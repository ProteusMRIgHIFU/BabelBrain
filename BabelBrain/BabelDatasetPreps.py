'''
Tools to generate files for acoustic/viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 23, 2021
     last update   - Sep 30, 2022

'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import os
import trimesh
import nibabel
from nibabel import processing
from nibabel.affines import AffineError, to_matvec
from nibabel.imageclasses import spatial_axes_first
from nibabel.nifti1 import Nifti1Image
from scipy import ndimage
from trimesh import creation 
import pymeshfix
from scipy.spatial.transform import Rotation as R
from skimage.measure import label, regionprops
import vtk
import pyvista as pv
import time
import gc
import yaml
from histoprint import text_hist, print_hist
import pandas as pd
import platform
import sys
from linetimer import CodeTimer
import re
from glob import glob
from pathlib import Path
import tempfile
import subprocess

try:
    import CTZTEProcessing
except:
    from . import CTZTEProcessing


try:
    from ConvMatTransform import ReadTrajectoryBrainsight, GetIDTrajectoryBrainsight,read_itk_affine_transform,itk_to_BSight
except:
    from .ConvMatTransform import ReadTrajectoryBrainsight, GetIDTrajectoryBrainsight,read_itk_affine_transform,itk_to_BSight

try:
    from FileManager import FileManager
except:
    from .FileManager import FileManager

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

def smooth(inputModel, method='Laplace', iterations=30, laplaceRelaxationFactor=0.5, taubinPassBand=0.1, boundarySmoothing=True):
    """Smoothes surface model using a Laplacian filter or Taubin's non-shrinking algorithm.
    """
    if method == "Laplace":
      smoothing = vtk.vtkSmoothPolyDataFilter()
      smoothing.SetRelaxationFactor(laplaceRelaxationFactor)
    else:  # "Taubin"
      smoothing = vtk.vtkWindowedSincPolyDataFilter()
      smoothing.SetPassBand(taubinPassBand)
    smoothing.SetBoundarySmoothing(boundarySmoothing)
    smoothing.SetNumberOfIterations(iterations)
    smoothing.SetInputData(inputModel)
    smoothing.Update()
    
    return smoothing.GetOutput()

def MaskToStl(binmask,affine):
    pvvol=pv.wrap(binmask.astype(np.float32))
    surface=pvvol.contour(isosurfaces=np.array([0.9]))
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        surface.save(tmpdirname+os.sep+'__t.vtk')

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(tmpdirname+os.sep+'__t.vtk')
        reader.ReadAllFieldsOn()
        reader.Update()

        writer = vtk.vtkSTLWriter()
        writer.SetInputData(smooth(reader.GetOutput()))
        writer.SetFileName(tmpdirname+os.sep+'__t.stl')
        writer.SetFileTypeToBinary()
        writer.Update()
        writer.Write()

        meshsurface=trimesh.load_mesh(tmpdirname+os.sep+'__t.stl')
        nP=(affine[:3,:3]@meshsurface.vertices.T).T
        nP[:,0]+=affine[0,3]
        nP[:,1]+=affine[1,3]
        nP[:,2]+=affine[2,3]
        meshsurface.vertices=nP

        os.remove(tmpdirname+os.sep+'__t.vtk')
        os.remove(tmpdirname+os.sep+'__t.stl')
    return meshsurface

MedianFilter=None
MedianCOMPUTING_BACKEND=''
def InitMedianGPUCallback(Callback=None,COMPUTING_BACKEND=2):
    global MedianFilter
    global MedianCOMPUTING_BACKEND
    MedianFilter = Callback
    if COMPUTING_BACKEND==1:
        MedianCOMPUTING_BACKEND='CUDA'
    elif COMPUTING_BACKEND==2:
        MedianCOMPUTING_BACKEND='OpenCL'
    elif COMPUTING_BACKEND==3:
        MedianCOMPUTING_BACKEND='Metal'
    else:
        MedianCOMPUTING_BACKEND='MLX'

VoxelizeFilter=None
VoxelizeCOMPUTING_BACKEND=''

def InitVoxelizeGPUCallback(Callback=None,COMPUTING_BACKEND=2):
    global VoxelizeFilter
    global VoxelizeCOMPUTING_BACKEND
    VoxelizeFilter = Callback
    if COMPUTING_BACKEND==1:
        VoxelizeCOMPUTING_BACKEND='CUDA'
    elif COMPUTING_BACKEND==2:
        VoxelizeCOMPUTING_BACKEND='OpenCL'
    elif COMPUTING_BACKEND==3:
        VoxelizeCOMPUTING_BACKEND='Metal'
    else:
        VoxelizeCOMPUTING_BACKEND='MLX'

MapFilter=None
MapFilterCOMPUTING_BACKEND=''
def InitMappingGPUCallback(Callback=None,COMPUTING_BACKEND=2):
    global MapFilter
    global MapFilterCOMPUTING_BACKEND
    MapFilter = Callback
    if COMPUTING_BACKEND==1:
        MapFilterCOMPUTING_BACKEND='CUDA'
    elif COMPUTING_BACKEND==2:
        MapFilterCOMPUTING_BACKEND='OpenCL'
    elif COMPUTING_BACKEND==3:
        MapFilterCOMPUTING_BACKEND='Metal'
    else:
        MapFilterCOMPUTING_BACKEND='MLX'

ResampleFilter=None
ResampleFilterCOMPUTING_BACKEND=''
def InitResampleGPUCallback(Callback=None,COMPUTING_BACKEND=2):
    global ResampleFilter
    global ResampleFilterCOMPUTING_BACKEND
    ResampleFilter = Callback
    if COMPUTING_BACKEND==1:
        ResampleFilterCOMPUTING_BACKEND='CUDA'
    elif COMPUTING_BACKEND==2:
        ResampleFilterCOMPUTING_BACKEND='OpenCL'
    elif COMPUTING_BACKEND==3:
        ResampleFilterCOMPUTING_BACKEND='Metal'
    else:
        ResampleFilterCOMPUTING_BACKEND='MLX'

BinaryClosingFilter=None
BinaryClosingFilterCOMPUTING_BACKEND=''
def InitBinaryClosingGPUCallback(Callback=None,COMPUTING_BACKEND=2):
    global BinaryClosingFilter
    global BinaryClosingFilterCOMPUTING_BACKEND
    BinaryClosingFilter = Callback
    if COMPUTING_BACKEND==1:
        BinaryClosingFilterCOMPUTING_BACKEND='CUDA'
    elif COMPUTING_BACKEND==2:
        BinaryClosingFilterCOMPUTING_BACKEND='OpenCL'
    elif COMPUTING_BACKEND==3:
        BinaryClosingFilterCOMPUTING_BACKEND='Metal'
    else:
        BinaryClosingFilterCOMPUTING_BACKEND='MLX'

LabelImage=None
LabelImageCOMPUTING_BACKEND=''
def InitLabelImageGPUCallback(Callback=None,COMPUTING_BACKEND=2):
    global LabelImage
    global LabelImageCOMPUTING_BACKEND
    LabelImage = Callback
    if COMPUTING_BACKEND==1:
        LabelImageCOMPUTING_BACKEND='CUDA'
    elif COMPUTING_BACKEND==2:
        LabelImageCOMPUTING_BACKEND='OpenCL'
    elif COMPUTING_BACKEND==3:
        LabelImageCOMPUTING_BACKEND='Metal'
    else:
        LabelImageCOMPUTING_BACKEND='MLX'

def ConvertMNItoSubjectSpace(M1_C,DataPath,T1Conformal_nii,bUseFlirt=True,PathSimnNIBS=''):
    '''
    Convert MNI coordinates to patient coordinates using SimbNIBS converted data
    
    M1_C is a [3] list or array of 3 MNI coordinates
    DataPath is the path to the main subject data directory where the XXXX_T1fs_conform.nii.gz SimbNIBS T1W output file is located
    T1Conformal_nii is the name of SimbNIBS T1W output file, such as SimbNIBS_LIFU_01_T1fs_conform.nii.gz
    
    bUseFlirt (default True) indicates if using FSL flirt as the tool to convert from MNI to subject coordinates, otherwise use 
    PathSimnNIBS path to mni2subject_coords SimnNIBS tool

    ABOUT:
         author        - Samuel Pichardo
         date          - Nov 27, 2021
         last update   - Nov 27, 2021
    
    '''
    if bUseFlirt:
        M1_MNI = '%f %f %f' % (M1_C[0],M1_C[1],M1_C[2])
        with open(DataPath+'mni.csv','w') as f:
            f.write(M1_MNI)
        cmd='$FSLDIR/bin/flirt -in "' + T1Conformal_nii + '" -ref $FSLDIR/data/standard/MNI152_T1_1mm -omat "'+\
            DataPath+'anat2mni.xfm" -out "' + DataPath+ 'anat_norm"'
        print(cmd)
        res=os.system(cmd)
        if res !=0:
            raise SystemError("Something didn't work when trying to run flirt")

        cmd ="$FSLDIR/bin/std2imgcoord -img '" + T1Conformal_nii +"' -std $FSLDIR/data/standard/MNI152_T1_1mm.nii -xfm '"+\
            DataPath+"anat2mni.xfm' '"+ DataPath+"mni.csv' > '" + DataPath+"natspace.tsv'"
        print(cmd)
        res=os.system(cmd)
        if res !=0:
            raise ValueError("Something didn't work when trying to convert from MNI to subject, check all paths are correct")

        with open(DataPath+'natspace.tsv','r') as f:
            subjectcoordinates=f.readlines()
        subjectcoordinates=np.asarray(subjectcoordinates[0].split('  ')).astype(float)

    else:
        M1_MNI = 'Generic,%f,%f,%f,M1'
        with open(DataPath+'mni.csv','w') as f:
            f.write(M1_MNI)
        cmd = PathSimnNIBS+"mni2subject_coords -m '"+DataPath+"m2m_SimbNIBS_LIFU_02/' -s '"+DataPath+"mni.csv'" +\
          " -o '" +DataPath+"subject.csv'"
        print(cmd)
        res=os.system(cmd)
        if res !=0:
            raise ValueError("Something didn't work when trying to convert from MNI to subject, check all paths are correct")
        with open(DataPath+'subject.csv','r') as f:
            subjectcoordinates=f.readlines()
        subjectcoordinates=np.asarray(subjectcoordinates[0].split(',')[1:4]).astype(np.float)
    print('MNI coordinates',M1_C)
    print('patient coordinates',subjectcoordinates)
    return subjectcoordinates

def DoIntersect(Mesh1,Mesh2,bForceUseBlender=False):
    # Fix broken meshes
    if Mesh1.body_count != 1:
        print('Mesh 1 is invalid... trying to fix')
        Mesh1 = FixMesh(Mesh1)
    if Mesh2.body_count != 1:
        print('Mesh 2 is invalid... trying to fix')
        Mesh2 = FixMesh(Mesh2)
    # Perform intersection
    if not bForceUseBlender:
        Mesh1_intersect =trimesh.boolean.intersection((Mesh1,Mesh2),engine='manifold')
    else:
        Mesh1_intersect =trimesh.boolean.intersection((Mesh1,Mesh2),engine='blender')

    # Check intersection is valid
    if Mesh1_intersect.is_empty:
        raise ValueError("Trajectory is outside headspace")    
    
    return Mesh1_intersect

def FixMesh(inmesh):
    with tempfile.TemporaryDirectory() as tmpdirname:
        inmesh.export(tmpdirname+os.sep+'__in.stl')
        pymeshfix.clean_from_file(tmpdirname+os.sep+'__in.stl', tmpdirname+os.sep+'__out.stl')
        fixmesh=trimesh.load_mesh(tmpdirname+os.sep+'__out.stl')
        os.remove(tmpdirname+os.sep+'__in.stl')
        os.remove(tmpdirname+os.sep+'__out.stl')
    return fixmesh

def RunMeshConv(reference,mesh,finalname,SimbNINBSRoot=''):
    scriptbase=os.path.join(resource_path(),"ExternalBin/SimbNIBSMesh/")
    if sys.platform == 'linux' or _IS_MAC:
        if sys.platform == 'linux':
            shell='bash'
            path_script = os.path.join(resource_path(),"ExternalBin/SimbNIBSMesh/run_linux.sh")
        elif _IS_MAC:
            shell='zsh'
            path_script = os.path.join(resource_path(),"ExternalBin/SimbNIBSMesh/run_mac.sh")
        
        print("Starting MeshConv")
        if _IS_MAC:
            cmd ='source "'+path_script + '" "' + SimbNINBSRoot + '" "' + scriptbase + '" "' + reference + '" "' + mesh +'" "' + finalname + '"'
            print(cmd)
            result = os.system(cmd)
        else:
            result = subprocess.run(
                    [shell,
                    path_script,
                    SimbNINBSRoot,
                    reference,
                    mesh,
                    finalname], capture_output=True, text=True
            )
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            result=result.returncode 
    else:
        path_script = os.path.join(resource_path(),"ExternalBin/SimbNIBSMesh/run_win.bat")
        
        print("Starting MeshConv")
        result = subprocess.run(
                [path_script,
                SimbNINBSRoot,
                reference,
                mesh,
                finalname], capture_output=True, text=True,shell=True,
        )
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        result=result.returncode 
    print("MeshConv Finished")
    
        
#process first with SimbNIBS
def GetSkullMaskFromSimbNIBSSTL(SimbNIBSDir='4007/4007_keep/m2m_4007_keep/',
                                SimbNIBSType='charm',# indicate if processing was done with charm or headreco
                                T1Source_nii ='4007/4007_keep/m2m_4007_keep/T1.nii.gz',
                                T1Conformal_nii='4007/4007_keep/m2m_4007_keep/T1fs_conform.nii.gz', #be sure it is the conformal 
                                CT_or_ZTE_input=None,
                                CTType = 0,
                                CoregCT_MRI=0, #if using CT, 0 does not coreg (assuming this was done previously), 1 from CT to MRI
                                ZTERange=(0.1,0.6),
                                HUThreshold=800.0,
                                HUCapThreshold=2100.0,
                                CT_quantification=10, #bits
                                Mat4Trajectory=None, 
                                TrajectoryType='brainsight',                               
                                Foc=135.0, #Tx focal length
                                TxDiam=157.0, # Tx aperture diameter used for FOV subvolume
                                Location=[27.5, -42, 42],#RAS location of target ,
                                TrabecularProportion=0.8, #proportion of trabecular bone
                                SpatialStep=1500/500e3/9*1e3, #step of mask to reconstruct , mm
                                prefix='', #Id to add to output file for identification
                                bPlot=True,
                                bForceFullRecalculation=False,
                                bForceUseBlender=False, # we use manifol3d by default, but we let open to use blender as backup
                                factorEnlargeRadius=1.05,
                                bApplyBOXFOV=False,
                                FOVDiameter=60.0, # diameter for  manual FOV
                                FOVLength=300.0, # lenght FOV
                                ElastixOptimizer='AdaptiveStochasticGradientDescent',
                                bDisableCTMedianFilter=False,
                                PetraMRIPeakDistance=50,
                                PetraNPeaks=2,
                                bInvertZTE=False,
                                bGeneratePETRAHistogram=False,
                                bSegmentBrainTissue=False,
                                SimbNINBSRoot='',
                                PETRASlope=-2929.6,
                                PETRAOffset=3274.9,
                                ZTESlope=-2085.0,
                                ZTEOffset=2329.0,
                                DensityThreshold=1200.0, #this is in case the input data is rather a density map
                                DeviceName='',
                                bMaximizeBoneRim=False,
                                bSaveCTMaximized=False): #created reduced FOV
    '''
    Generate masks for acoustic/viscoelastic simulations. 
    It creates an Nifti file that is in subject space using as main inputs the output files of the headreco tool and location of coordinates where focal point is desired
    
    
    ABOUT:
         author        - Samuel Pichardo
         date          - June 23, 2021
         last update   - Nov 27, 2021
    
    '''
    print('Starting Masking Process')
    
    if CTType==4:
        TypeThresold=DensityThreshold
    else:
        TypeThresold=HUThreshold

    # Create file manager for step 1
    S1_file_manager = FileManager(simNIBS_dir = SimbNIBSDir,
                                    simbNIBS_type = SimbNIBSType,
                                    T1_fname = T1Source_nii,
                                    T1_iso_fname = T1Conformal_nii,
                                    extra_scan_fname = CT_or_ZTE_input,
                                    prefix = prefix,
                                    current_CT_type = CTType,
                                    coreg = CoregCT_MRI,
                                    current_HUT = TypeThresold,
                                    current_pCT_range = ZTERange)

    inputfilenames = S1_file_manager.input_files
    outputfilenames = S1_file_manager.output_files

    if bSegmentBrainTissue:
        mshfile = glob(os.path.join(SimbNIBSDir,'*.msh'))
        if len(mshfile)!=1:
            raise RuntimeError("There should be one (and only one) .msh file at " + SimbNIBSDir)
        mshfile=mshfile[0]

    #load T1W
    T1Conformal = S1_file_manager.load_file(T1Conformal_nii)
    baseaffine=T1Conformal.affine.copy()
    print('baseaffine',baseaffine,"\n")
    #sanity test to verify we have an isotropic scan
    assert(np.allclose(np.array(T1Conformal.header.get_zooms()),np.ones(3),rtol=1e-3))
    
    baseaffine[0,0]*=SpatialStep
    baseaffine[1,1]*=SpatialStep
    baseaffine[2,2]*=SpatialStep

    skull_stl=outputfilenames['Skull_STL']
    csf_stl=outputfilenames['CSF_STL']
    skin_stl=outputfilenames['Skin_STL']
    
    if bForceFullRecalculation==False:
        with CodeTimer("Checking if previously generated files can be reused", unit="s"):
            bReuseFiles, prevoutputfilenames = S1_file_manager.check_reuse_files(inputfilenames,outputfilenames)
    else:
        bReuseFiles=False
    print(f"bReuseFiles, bForceFullRecalculation: {bReuseFiles,bForceFullRecalculation}")

    if SimbNIBSType=='charm':
        if bReuseFiles:
            tissues = S1_file_manager.load_file(inputfilenames['SimbNIBSinput'])
            tissues = S1_file_manager.nibabel_to_sitk(tissues)
            TMaskItk = tissues > 0
        else:
            #while charm is much more powerful to segment skull regions, we need to calculate the meshes ourselves
            charminput = inputfilenames['SimbNIBSinput']
            charm = S1_file_manager.load_file(charminput)
            charmdata=np.ascontiguousarray(charm.get_fdata())[:,:,:,0]
            AllTissueRegion=charmdata>0 #this mimics what the old headreco does for skin
            
            tissues = S1_file_manager.load_file(charminput)
            tissues = S1_file_manager.nibabel_to_sitk(tissues)
            TMaskItk = tissues > 0

            BoneRegion=(charmdata>0) & (charmdata!=5) #this mimics what the old headreco does for bone
            CSFRegion=(charmdata==1) | (charmdata==2) | (charmdata==3) | (charmdata==9) #this mimics what the old headreco does for skin
            with CodeTimer("charm surface recon",unit='s'):
                skin_mesh=MaskToStl(AllTissueRegion,charm.affine)
                csf_mesh=MaskToStl(CSFRegion,charm.affine)
                skull_mesh=MaskToStl(BoneRegion,charm.affine)
                
                if skin_mesh.body_count != 1:
                    print('skin_mesh is invalid... trying to fix')
                    skin_mesh = FixMesh(skin_mesh)
                S1_file_manager.save_file(file_data=skin_mesh,filename=skin_stl)

                if csf_mesh.body_count != 1:
                    print('csf_mesh is invalid... trying to fix')
                    csf_mesh = FixMesh(csf_mesh)
                S1_file_manager.save_file(file_data=csf_mesh,filename=csf_stl)

                if skull_mesh.body_count != 1:
                    print('skull_mesh is invalid... trying to fix')
                    skull_mesh = FixMesh(skull_mesh)
                S1_file_manager.save_file(file_data=skull_mesh,filename=skull_stl)

            S1_file_manager.save_file(file_data=None,filename=outputfilenames['ReuseSimbNIBS'],precursor_files=inputfilenames.values())
    else:
        tissues = S1_file_manager.load_file(inputfilenames['SimbNIBSinput'])
        tissues = S1_file_manager.nibabel_to_sitk(tissues)
        TMaskItk = tissues > 0
        S1_file_manager.save_file(file_data=None,filename=outputfilenames['ReuseSimbNIBS'],precursor_files=inputfilenames.values())

    #building a cone object representing acoustic beam pointing to desired location
    RadCone=TxDiam/2*factorEnlargeRadius
    if type(Foc) is tuple:
        Foc=Foc[0]
    HeightCone=np.sqrt(Foc**2-RadCone**2)
    print('HeightCone',HeightCone)
    
    InVAffine=np.linalg.inv(baseaffine)

    if TrajectoryType =='brainsight':
        print('*'*40+'\n Reading orientation and target location directly from Brainsight export\n'+'*'*40)
        RMat=ReadTrajectoryBrainsight(Mat4Trajectory)
    else:
        inMat=read_itk_affine_transform(Mat4Trajectory)
         #we add this as in Brainsight the needle for trajectory starts at with a vector pointing 
         #to the feet direction , while in SlicerIGT it starts with a vector towards the head
        print('*'*40+'\n Reading orientation and target location directly from Slicer export\n'+'*'*40)
        RMat = itk_to_BSight(inMat)

    print('Trajectory Matrix\n',RMat)
    Location=RMat[:3,3].tolist()
    print('Location',Location)
    #Ok maybe a bit too lacking of simplification..... 
    TransformationCone=np.eye(4)
    TransformationCone[2,2]=-1
    OrientVec=np.array([0,0,1]).reshape((1,3))
    TransformationCone[0,3]=Location[0]
    TransformationCone[1,3]=Location[1]
    TransformationCone[2,3]=Location[2]+HeightCone
    Cone=creation.cone(RadCone,HeightCone)
    Cone.apply_transform(TransformationCone.copy())

    TransformationCone[2,3]=Location[2]
   
    TransformationCone=np.eye(4)
    TransformationCone[0,3]=-Location[0]
    TransformationCone[1,3]=-Location[1]
    TransformationCone[2,3]=-Location[2]

    Cone.apply_transform(TransformationCone)
    
    RMat=RMat[:3,:3]
    
    TransformationCone=np.eye(4)
    TransformationCone[0,3]=Location[0]
    TransformationCone[1,3]=Location[1]
    TransformationCone[2,3]=Location[2]
    
    TransformationCone[0:3,0:3]=RMat

    Cone.apply_transform(TransformationCone)

    CumulativeTransform=TransformationCone.copy()

    print('Final RMAT')
    print(RMat)
    
    skin_mesh = S1_file_manager.load_file(skin_stl)
    #we intersect the skin region with a cone region oriented in the same direction as the acoustic beam
    skin_mesh =DoIntersect(skin_mesh,Cone,bForceUseBlender=bForceUseBlender)

    #we obtain the list of Cartesian voxels inside the skin region intersected by the cone    
    with CodeTimer("voxelization ",unit='s'):
        if VoxelizeFilter is None:  
            skin_grid = skin_mesh.voxelized(SpatialStep,max_iter=30).fill().points
        else:
            skin_grid = VoxelizeFilter(skin_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)

    
    x_vec=np.arange(skin_grid[:,0].min(),skin_grid[:,0].max()+SpatialStep,SpatialStep)
    y_vec=np.arange(skin_grid[:,1].min(),skin_grid[:,1].max()+SpatialStep,SpatialStep)
    z_vec=np.arange(skin_grid[:,2].min(),skin_grid[:,2].max()+SpatialStep,SpatialStep)
    
    Corner1=np.array([x_vec[0],y_vec[0],z_vec[0],1]).reshape((4,1))
    Corner2=np.array([x_vec[-1],y_vec[-1],z_vec[-1],1]).reshape((4,1))

    #we will produce one dataset (used only for sanity tests)
    # with the same orientation as the T1 scan and just enclosing the list of points intersected
    AffIJKCorners=np.floor(np.dot(InVAffine,np.hstack((Corner1,Corner2)))).astype(np.int64).T
    AffIJKCornersMin=np.min(AffIJKCorners,axis=0).reshape((4,1))
    NewOrig=np.dot(baseaffine,AffIJKCornersMin)
    
    baseaffine[:,3]=NewOrig.flatten()
    
    
    print('baseaffine',baseaffine)
    RMat4=np.eye(4)
    RMat4[:3,:3]=RMat*SpatialStep
    print('RMat4',RMat4)
   
    baseaffineRot=RMat4   
    
    
    print('baseaffineRot',baseaffineRot)
    
    InVAffine=np.linalg.inv(baseaffine)
    InVAffineRot=np.linalg.inv(baseaffineRot)

    XYZ=skin_grid
    #we make it a Nx4 array
    XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1)))).T
    
    #now we prepare the new dataset that is perpendicular to the cone direction
    #first we calculate the indexes i,j,k
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int64).T
    NewOrig=baseaffineRot @np.array([AffIJK[:,0].min(),AffIJK[:,1].min(),AffIJK[:,2].min(),1]).reshape((4,1))
    baseaffineRot[:,3]=NewOrig.flatten()
    InVAffineRot=np.linalg.inv(baseaffineRot)
    
    ALoc=np.ones((4,1))
    ALoc[:3,0]=np.array(Location)
    XYZ=np.hstack((XYZ,ALoc))
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int64).T
    
    LocFocalPoint=AffIJK[-1,:3] #we recover the location in pixels of the intended target
          
    ## This is the box that covers the minimal volume
    DimsBox=np.zeros((3))
    DimsBox[0:2]=FOVDiameter
    DimsBox[2]+=FOVLength
    TransformationBox=np.eye(4)

    BoxFOV=creation.box(DimsBox,
                        transform=TransformationBox)
    BoxFOV.apply_transform(CumulativeTransform)
    S1_file_manager.save_file(file_data=BoxFOV,filename=os.path.dirname(T1Conformal_nii)+os.sep+prefix+'_box_FOV.stl')
      
    ##################### And we repeat and complete data extraction
    baseaffine=T1Conformal.affine.copy()
    baseaffine[0,0]*=SpatialStep
    baseaffine[1,1]*=SpatialStep
    baseaffine[2,2]*=SpatialStep

    skull_mesh = S1_file_manager.load_file(skull_stl)
    csf_mesh = S1_file_manager.load_file(csf_stl)
    skin_mesh = S1_file_manager.load_file(skin_stl)

    if bApplyBOXFOV:
        skull_mesh=DoIntersect(skull_mesh,BoxFOV,bForceUseBlender=bForceUseBlender)
        csf_mesh  =DoIntersect(csf_mesh,  BoxFOV,bForceUseBlender=bForceUseBlender)
        skin_mesh =DoIntersect(skin_mesh, BoxFOV,bForceUseBlender=bForceUseBlender)
    
    #we first subtract to find the pure bone region
    if VoxelizeFilter is None:
        while(True):
            try:
                with CodeTimer("cpu skull voxelization",unit='s'):
                    skull_grid = skull_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                with CodeTimer("cpu voxelization",unit='s'):
                    csf_grid = csf_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                with CodeTimer("cpu voxelization",unit='s'):
                    skin_grid = skin_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                break
            except AttributeError as err:
                print("Repeating CSG boolean since once in while it returns an scene instead of a mesh....")
                print(err)
            else:
                raise err
    else:
        with CodeTimer("skull voxelization",unit='s'):
            skull_grid = VoxelizeFilter(skull_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)
        with CodeTimer("brain voxelization",unit='s'):
            csf_grid = VoxelizeFilter(csf_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)
        with CodeTimer("skin voxelization",unit='s'):
            skin_grid = VoxelizeFilter(skin_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)
        
    
    #we obtain the list of Cartesian voxels in the whole skin region intersected by the cone    
        
    x_vec=np.arange(skin_grid[:,0].min(),skin_grid[:,0].max()+SpatialStep,SpatialStep)
    y_vec=np.arange(skin_grid[:,1].min(),skin_grid[:,1].max()+SpatialStep,SpatialStep)
    z_vec=np.arange(skin_grid[:,2].min(),skin_grid[:,2].max()+SpatialStep,SpatialStep)
    
    Corner1=np.array([x_vec[0],y_vec[0],z_vec[0],1]).reshape((4,1))
    Corner2=np.array([x_vec[-1],y_vec[-1],z_vec[-1],1]).reshape((4,1))
    
    #we will produce one dataset (used only for sanity tests)
    # with the same orientation as the T1 scan and just enclosing the list of points intersected
    AffIJKCorners=np.floor(np.dot(InVAffine,np.hstack((Corner1,Corner2)))).astype(np.int64).T
    AffIJKCornersMin=np.min(AffIJKCorners,axis=0).reshape((4,1))
    NewOrig=np.dot(baseaffine,AffIJKCornersMin)
    
    baseaffine[:,3]=NewOrig.flatten()
    
    
    print('baseaffine',baseaffine)
    baseaffineRot=baseaffine.copy()
    RMat4=np.eye(4)
    RMat4[:3,:3]=RMat*SpatialStep
    print('RMat4',RMat4)
    
    
    baseaffineRot=RMat4
    
    print('baseaffineRot',baseaffineRot)
    
    InVAffine=np.linalg.inv(baseaffine)
    InVAffineRot=np.linalg.inv(baseaffineRot)
    InVAffineRot=InVAffineRot.astype(skin_grid.dtype)    

    with CodeTimer("skin masking",unit='s'):
        XYZ=skin_grid
        XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1),dtype=skin_grid.dtype))).T
        #now we prepare the new dataset that is perpendicular to the cone direction
        AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int64).T
        NewOrig=baseaffineRot @np.array([AffIJK[:,0].min(),AffIJK[:,1].min(),AffIJK[:,2].min(),1]).reshape((4,1))
        baseaffineRot[:,3]=NewOrig.flatten()
        InVAffineRot=np.linalg.inv(baseaffineRot)
        InVAffineRot=InVAffineRot.astype(skin_grid.dtype)
        del AffIJK
        ALoc=np.ones((4,1),dtype=skin_grid.dtype)
        ALoc[:3,0]=np.array(Location,dtype=skin_grid.dtype)
        XYZ=np.hstack((XYZ,ALoc))
        AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int64).T
        
        LocFocalPoint=AffIJK[-1,:3]
        
        BinMaskConformalSkinRot=np.zeros(np.max(AffIJK,axis=0)[:3]+1,np.int8)
        BinMaskConformalSkinRot[AffIJK[:,0],AffIJK[:,1],AffIJK[:,2]]=1

    del XYZ
    del skin_grid
    gc.collect()

    t0=time.time()
    with CodeTimer("skull masking",unit='s'):
        XYZ=skull_grid
        XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1),dtype=skull_grid.dtype))).T
        AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int64).T
        BinMaskConformalSkullRot=np.zeros_like(BinMaskConformalSkinRot)
        inds=(AffIJK[:,0]<BinMaskConformalSkullRot.shape[0])&\
             (AffIJK[:,1]<BinMaskConformalSkullRot.shape[1])&\
             (AffIJK[:,2]<BinMaskConformalSkullRot.shape[2])
        AffIJK=AffIJK[inds,:]
        BinMaskConformalSkullRot[AffIJK[:,0],AffIJK[:,1],AffIJK[:,2]]=1
    del AffIJK
    del XYZ
    del skull_grid
    gc.collect()

    with CodeTimer("csf masking",unit='s'):
        XYZ=csf_grid
        XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1),dtype=csf_grid.dtype))).T
        AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int64).T
        BinMaskConformalCSFRot=np.zeros(BinMaskConformalSkinRot.shape,np.uint8)
        inds=(AffIJK[:,0]<BinMaskConformalSkullRot.shape[0])&\
             (AffIJK[:,1]<BinMaskConformalSkullRot.shape[1])&\
             (AffIJK[:,2]<BinMaskConformalSkullRot.shape[2])
        AffIJK=AffIJK[inds,:]
        BinMaskConformalCSFRot[AffIJK[:,0],AffIJK[:,1],AffIJK[:,2]]=1
   
    del AffIJK
    del XYZ
    del csf_grid
    gc.collect()
    
    FinalMask=BinMaskConformalSkinRot

    #Now we deal if CT or ZTE has beegn given as input
    if CT_or_ZTE_input is  None:
        with CodeTimer("Final Mask gen without CT/ZTE", unit="s"):
            FinalMask[BinMaskConformalSkullRot==1]=2 #cortical
            FinalMask[BinMaskConformalCSFRot==1]=4#brain
    else:
        if bReuseFiles:
            # Grab previously generated mask
            fct = S1_file_manager.load_file(prevoutputfilenames['ReuseMask'])

            # Load in appropriate rCT and resave fct under current file name
            if CTType in [2,3]:
                rCT = S1_file_manager.load_file(outputfilenames['pCTfname'])
                S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=outputfilenames['pCTfname'])
            else:
                if CoregCT_MRI == 0:
                    rCT = S1_file_manager.load_file(inputfilenames['ExtraScan'])
                    S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=[outputfilenames['ReuseSimbNIBS'],outputfilenames['Skull_STL'],outputfilenames['CSF_STL'],outputfilenames['Skin_STL']])
                elif CoregCT_MRI == 1:
                    rCT = S1_file_manager.load_file(outputfilenames['CTInT1W'])
                    S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=outputfilenames['CTInT1W'])
                else:
                    rCT = S1_file_manager.load_file(outputfilenames['T1WinCT'])
                    S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=outputfilenames['T1WinCT'])

            rCTdata=rCT.get_fdata()
        else:
            if CTType in [2,3]:
                print('Processing ZTE/PETRA to pCT')
                bIsPetra = CTType==3
                with CodeTimer("Bias and coregistration ZTE/PETRA to T1",unit='s'):
                    rT1,rZTE = CTZTEProcessing.BiasCorrecAndCoreg(T1Conformal_nii,
                                                                  TMaskItk,
                                                                  S1_file_manager,
                                                                  ElastixOptimizer,
                                                                  bIsPetra=bIsPetra,
                                                                  bInvertZTE=bInvertZTE)
                with CodeTimer("Conversion ZTE/PETRA to pCT",unit='s'):
                    rCT = CTZTEProcessing.ConvertZTE_PETRA_pCT(rT1,
                                                               rZTE,
                                                               TMaskItk,
                                                               S1_file_manager,
                                                               bIsPetra=bIsPetra,
                                                               PetraMRIPeakDistance=PetraMRIPeakDistance,
                                                               PetraNPeaks=PetraNPeaks,
                                                               bGeneratePETRAHistogram=bGeneratePETRAHistogram,
                                                               PETRASlope=PETRASlope,
                                                               PETRAOffset=PETRAOffset,
                                                               ZTESlope=ZTESlope,
                                                               ZTEOffset=ZTEOffset)
            else:
                with CodeTimer("Coregistration CT to T1",unit='s'):
                    rCT = CTZTEProcessing.CTCorreg(T1Conformal_nii,
                                                   S1_file_manager,
                                                   ElastixOptimizer,
                                                   ResampleFilter,
                                                   ResampleFilterCOMPUTING_BACKEND)
            rCTdata=rCT.get_fdata()
            hist = np.histogram(rCTdata[rCTdata>TypeThresold],bins=15)
            print('*'*40)
            if CTType in [1,2,3]:
                title = "CT HU"
            else:
                title= "Density"
            print_hist(hist, title=title, symbols=r"=",fg_colors="0",bg_colors="0",columns=80)
            if CTType in [1,2,3]:
                rCTdata[rCTdata>HUCapThreshold]=HUCapThreshold #we threshold only CT-type data

            fct=nibabel.Nifti1Image((rCTdata>TypeThresold).astype(np.float32), affine=rCT.affine)

            if CTType in [2,3]:
                S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=outputfilenames['pCTfname'])
            else:
                if CoregCT_MRI == 0:
                    S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=[outputfilenames['ReuseSimbNIBS'],outputfilenames['Skull_STL'],outputfilenames['CSF_STL'],outputfilenames['Skin_STL']])
                elif CoregCT_MRI == 1:
                    S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=outputfilenames['CTInT1W'])
                else:
                    S1_file_manager.save_file(file_data=fct,filename=outputfilenames['ReuseMask'],precursor_files=outputfilenames['T1WinCT'])            

        mask_nifti2 = nibabel.Nifti1Image(FinalMask, affine=baseaffineRot)

        ############
        with CodeTimer("CT extrapol",unit='s'):
            print('rCTdata range',rCTdata.min(),rCTdata.max())
            rCT = nibabel.Nifti1Image(rCTdata, rCT.affine, rCT.header)
            if ResampleFilter is None:
                nCT=processing.resample_from_to(rCT,mask_nifti2,mode='constant',cval=rCTdata.min())
            else:
                nCT=ResampleFilter(rCT,mask_nifti2,mode='constant',cval=rCTdata.min(),GPUBackend=ResampleFilterCOMPUTING_BACKEND)

            RatioCTVoxels=np.ceil(2.0/np.array(nCT.header.get_zooms())).astype(int) # 1 mm distance

            ndataCT=np.ascontiguousarray(nCT.get_fdata()).astype(np.float32)
            if CTType in [1,2,3]:
                ndataCT[ndataCT>HUCapThreshold]=HUCapThreshold
            print('ndataCT range',ndataCT.min(),ndataCT.max())
            
            if not bDisableCTMedianFilter:
                with CodeTimer("median filter CT/Density",unit='s'):
                    print('Theshold for bone',TypeThresold)
                    if MedianFilter is None:
                        fct=ndimage.median_filter(ndataCT>TypeThresold,7,mode='constant',cval=0)
                    else:
                        fct=MedianFilter(np.ascontiguousarray(ndataCT>TypeThresold).astype(np.uint8),7,GPUBackend=MedianCOMPUTING_BACKEND)
            else:
                fct = ndataCT>TypeThresold
            sf2=np.round((np.ones(3)*5)/mask_nifti2.header.get_zooms()).astype(int)
            with CodeTimer("binary closing CT/Density",unit='s'):
                fct = BinaryClosingFilter(fct, structure=np.ones(sf2,dtype=int), GPUBackend=BinaryClosingFilterCOMPUTING_BACKEND)
            nfct=fct!=0

            with CodeTimer("label CT",unit='s'):
                if LabelImage is None:
                    label_img=label(nfct)
                else:
                    label_img = LabelImage(nfct, GPUBackend=LabelImageCOMPUTING_BACKEND)
                regions= regionprops(label_img)
                regions=sorted(regions,key=lambda d: d.area)
                nfct=label_img==regions[-1].label

            ndataCT[nfct==False]=0

        with CodeTimer("CT/Density binary_dilation",unit='s'):
            BinMaskConformalCSFRot= ndimage.binary_dilation(BinMaskConformalCSFRot,iterations=6)
        with CodeTimer("FinalMask[BinMaskConformalCSFRot]=4",unit='s'):
            FinalMask[BinMaskConformalCSFRot]=4  
            FinalMask[BinMaskConformalSkullRot==1]=4
        #brain
        with CodeTimer("FinalMask[nfct]=2",unit='s'):
            FinalMask[nfct]=2  #bone
        #we do a cleanup of islands of skin that are isolated between the skull region and brain region, we assume all except the largest one are brain tissue
        with CodeTimer("Labeling",unit='s'):
            if LabelImage is None:
                label_img = label(FinalMask==1)
            else:
                label_img = LabelImage(FinalMask==1, GPUBackend=LabelImageCOMPUTING_BACKEND)
        
        with CodeTimer("regionprops",unit='s'):
            regions= regionprops(label_img)

        print("number of skin region islands", len(regions))
        regions=sorted(regions,key=lambda d: d.area)
        AllLabels=[]
        for l in regions[:-1]:
            if bApplyBOXFOV is  False:
                AllLabels.append(l.label)
            else:
                if l.area < 0.1*regions[-1].area: #in subvolumes, often the far field region is bigger than the near field
                    AllLabels.append(l.label)

        FinalMask[np.isin(label_img,np.array(AllLabels))]=4
            
        CTBone=ndataCT[nfct]
        CTBone[CTBone<TypeThresold]=TypeThresold #we cut off to avoid problems in acoustic sim
        ndataCT[nfct]=CTBone

        if bMaximizeBoneRim:
            with CodeTimer("Fixing partial volume artifacts edge",unit='s'):
                interior_high_th=800.0
                max_boost = 1000.0
                nPixelsErode=RatioCTVoxels.copy()
                print('nPixelsErode',nPixelsErode)
                #we scan 1mm around the edge
                for ntr in range(len(RatioCTVoxels)):
                    if RatioCTVoxels[ntr]%2==0:
                        RatioCTVoxels[ntr]+=1
                    if RatioCTVoxels[ntr]==1:
                        RatioCTVoxels[ntr]=3 #minimum 3 voxels
    
                interior_mask=nfct.copy().astype(np.uint8)

                # # # Create conservative interior bone mask (higher threshold + erosion)
                interior_mask_val = (ndataCT >= interior_high_th)#.astype(np.uint8)
                with CodeTimer("binary erosion",unit='s'):
                    interior_mask = ndimage.binary_erosion(interior_mask,structure=np.ones(RatioCTVoxels),iterations=1)


                # # Precompute interior bone mean (global or local)
                global_interior_mean = ndataCT[interior_mask_val].mean()
                print("Global interior bone mean HU:", global_interior_mean)

                # distance transform from interior mask: for each voxel inside coarse mask,
                # compute distance to nearest interior voxel (in voxels)
                # We'll compute distance only within the nfct to save time
                # distance_to_interior: inside nfct -> distance to nearest interior voxel (0 if interior)
                inv_interior = 1 - interior_mask  # interior==1 => inv_interior==0; else 1
                # compute distance from every voxel to nearest interior voxel (Euclidean)
                with CodeTimer("distance_transform_edt",unit='s'):
                    dist_to_interior = ndimage.distance_transform_edt(inv_interior)  # voxels

                # # Identify edge voxels: in nfct but not in interior_mask
                edge_voxels = (nfct == 1) & (interior_mask == 0)

                # # For each edge voxel, compute weight based on distance (close -> high weight)
                # # weight = exp(-dist / distance_scale)  (so dist=0 => weight=1 ; dist large => ~0)
                distance_scale=float(RatioCTVoxels[0])/2.0
                dist = dist_to_interior[edge_voxels]
                weights = np.exp(-dist / distance_scale)

                # # Local approach: get a local interior mean per edge voxel by sampling interior voxels
                # # We'll compute a gaussian-blurred interior mean image for locality:
                interior_f = ndataCT * interior_mask_val  # interior intensity, zero elsewhere
                # # To get local mean of interior bone near each voxel, convolve with small gaussian and normalize by blurred mask
                sigma_local = RatioCTVoxels[0]  # small locality window in voxels (tune)
                with CodeTimer("interior_f gaussian_filter",unit='s'):
                    blur_interior = ndimage.gaussian_filter(interior_f, sigma=sigma_local)
                with CodeTimer("blur_mask gaussian_filter",unit='s'):
                    blur_mask = ndimage.gaussian_filter(interior_mask_val.astype(np.float32), sigma=sigma_local)
                # # avoid division by zero
                local_interior_mean_img = np.where(blur_mask > 1e-6, blur_interior / blur_mask, global_interior_mean)

                # # Now get local interior mean for each edge voxel
                local_means = local_interior_mean_img[edge_voxels]

                # # Compute corrected HU: blend original toward local interior mean using weight
                orig_vals = ndataCT[edge_voxels]
                correct_vals = orig_vals + weights * (local_means - orig_vals)

                # # Optionally clamp boost to avoid unrealistically large jumps
                delta = correct_vals - orig_vals
                delta_clipped = np.clip(delta, a_min=None, a_max=max_boost)  # only upper clamp
                correct_vals = orig_vals + delta_clipped

                CTnamefiltered=os.path.dirname(T1Conformal_nii)+os.sep+'CT_filtered.nii.gz'
                CTnamenonfiltered=os.path.dirname(T1Conformal_nii)+os.sep+'CT_nonfiltered.nii.gz'
                if bSaveCTMaximized:
                    with CodeTimer("saving CTnamenonfiltered",unit='s'):
                        nCTNifti=nibabel.Nifti1Image(ndataCT, nCT.affine, nCT.header)
                        nCTNifti.to_filename(CTnamenonfiltered)

                # ndataCT[nfct_rim]=CTBoneMaxFilter[nfct_rim]
                ndataCT[edge_voxels] = correct_vals

                if bSaveCTMaximized:
                    with CodeTimer("saving CTnamefiltered",unit='s'):
                        nCTNifti=nibabel.Nifti1Image(ndataCT, nCT.affine, nCT.header)
                        nCTNifti.to_filename(CTnamefiltered)

        maxData=ndataCT[nfct].max()
        minData=ndataCT[nfct].min()
        
        A=maxData-minData
        M = 2**CT_quantification-1
        ResStep=A/M 
        qx = ResStep *  np.round( (M/A) * (ndataCT[nfct]-minData) )+ minData
        ndataCT[nfct]=qx
        UniqueHU=np.unique(ndataCT[nfct])
        print('Unique CT/Density values',len(UniqueHU))
        CTCalfname = os.path.dirname(T1Conformal_nii)+os.sep+prefix+'CT-cal.npz'
        S1_file_manager.save_file(file_data=None,filename=CTCalfname,UniqueHU=UniqueHU)

        with CodeTimer("Mapping unique values",unit='s'):
            if MapFilter is None:
                ndataCTMap=np.zeros(ndataCT.shape,np.uint32)
                for n,d in enumerate(UniqueHU):
                    ndataCTMap[ndataCT==d]=n
                ndataCTMap[nfct==False]=0
            else:
                ndataCTMap=MapFilter(ndataCT,nfct.astype(np.uint8),UniqueHU,GPUBackend=MapFilterCOMPUTING_BACKEND)

            nCT=nibabel.Nifti1Image(ndataCTMap, nCT.affine, nCT.header)

            S1_file_manager.save_file(file_data=nCT,filename=outputfilenames['CTfname'],precursor_files=outputfilenames['ReuseMask'])

    with CodeTimer("final median filter ",unit='s'):
        if CT_or_ZTE_input is not None:
            FinalMask[FinalMask==2]=1
        if MedianFilter is None:
            FinalMask=ndimage.median_filter(FinalMask.astype(np.uint8),7)
        else:
            FinalMask=MedianFilter(FinalMask.astype(np.uint8),7,GPUBackend=MedianCOMPUTING_BACKEND)
    if CT_or_ZTE_input is not None:
        FinalMask[nfct]=2

        with CodeTimer("Second Labeling",unit='s'):
            if LabelImage is None:
                label_img = label(FinalMask==1)
            else:
                label_img = LabelImage(FinalMask==1, GPUBackend=LabelImageCOMPUTING_BACKEND)
        
        with CodeTimer("second regionprops",unit='s'):
            regions= regionprops(label_img)

        print("second number of skin region islands", len(regions))
        regions=sorted(regions,key=lambda d: d.area)
        AllLabels=[]
        for l in regions[:-1]:
            if bApplyBOXFOV is  False:
                AllLabels.append(l.label)
            else:
                if l.area < 0.1*regions[-1].area: #in subvolumes, often the far field region is bigger than the near field
                    AllLabels.append(l.label)

        FinalMask[np.isin(label_img,np.array(AllLabels))]=4

    #we extract back the bone part
    BinMaskConformalSkullRot=FinalMask==2
    LineViewBone=BinMaskConformalSkullRot[LocFocalPoint[0],LocFocalPoint[1],LocFocalPoint[2]:]
    #we erode to establish the section of bone associated with trabecular
    DegreeErosion=int(np.round(LineViewBone.sum()*(1-TrabecularProportion)))
    print('DegreeErosion',DegreeErosion)
    if DegreeErosion !=0:
        Trabecula=ndimage.binary_erosion(BinMaskConformalSkullRot,iterations=DegreeErosion)
        FinalMask[Trabecula]=3 #trabecula
    else:
        print('*************'*4)
        print('WARNING '*4)
        print('Conditions indicate that the whole bone region is considered trabecular.')
        print('Consider reducing fraction of trabecular bone in Advanced options.')
        print('*************'*4)
        
        #in case it is 0, it means all layer is trabeclar
        FinalMask[BinMaskConformalSkullRot]=3 #trabecula
    
    FinalMask[LocFocalPoint[0],LocFocalPoint[1],LocFocalPoint[2]]=5 #focal point location

    if CT_or_ZTE_input is not None:
        with CodeTimer("Final cleanup of prefocal region",unit='s'):
            FinalMask=np.flip(FinalMask,axis=2)
            Rloc = np.array(np.where(FinalMask==5)).flatten()
            for i in range(FinalMask.shape[0]):
                for j in range(FinalMask.shape[1]):
                    Line=FinalMask[i,j,:Rloc[2]]
                    bone = np.array(np.where((Line==2) | (Line==3))).flatten()
                    if len(bone)>0:
                        subline=Line[:bone.min()]
                        subline[subline==4]=1
                        FinalMask[i,j,:len(subline)]=subline
            
            FinalMask=np.flip(FinalMask,axis=2)

    if bSegmentBrainTissue:

        # we just need an empty file to pass matrx size and affine
        emptyNifti =   nibabel.Nifti1Image(FinalMask*0, affine=baseaffineRot)

        with CodeTimer("Upscaling final tissue to recover GM and WM masks",unit='s'):
            with tempfile.TemporaryDirectory() as tmpdirname:
                ename=os.path.join(tmpdirname,'empty.nii.gz')
                emptyNifti.to_filename(ename)
                outname = os.path.join(tmpdirname,'out.nii.gz')
                RunMeshConv(ename,mshfile,outname,SimbNINBSRoot=SimbNINBSRoot)
                upScaleMask=nibabel.load(outname).get_fdata().astype(np.int8)
        
        FinalMask2=FinalMask.copy()
        FinalMask2[upScaleMask==1]=6 #white matter
        FinalMask2[upScaleMask==2]=7 #gray matter
        FinalMask2[upScaleMask==3]=8 #CSF
        for n in range(6):
            if n!=4:
                FinalMask2[FinalMask==n]=n

        mask_nifti2 = nibabel.Nifti1Image(FinalMask2, affine=baseaffineRot) 
    else:
        mask_nifti2 = nibabel.Nifti1Image(FinalMask, affine=baseaffineRot)

    outname=os.path.dirname(T1Conformal_nii)+os.sep+prefix+'BabelViscoInput.nii.gz'
    S1_file_manager.save_file(file_data=mask_nifti2,filename=outname)
    
    with CodeTimer("resampling T1 to mask",unit='s'):
        if ResampleFilter is None:
            T1Conformal=processing.resample_from_to(T1Conformal,mask_nifti2,mode='constant',order=0,cval=T1Conformal.get_fdata().min())
        else:
            T1Conformal=ResampleFilter(T1Conformal,mask_nifti2,mode='constant',order=0,cval=T1Conformal.get_fdata().min(),GPUBackend=ResampleFilterCOMPUTING_BACKEND)
        T1W_resampled_fname=os.path.dirname(T1Conformal_nii)+os.sep+prefix+'T1W_Resampled.nii.gz'
        S1_file_manager.save_file(file_data=T1Conformal,filename=T1W_resampled_fname)
    
    if bPlot:
        plt.figure()
        plt.imshow(FinalMask[:,LocFocalPoint[1],:],cmap=plt.cm.jet)
        plt.gca().set_aspect(1.0)
        plt.colorbar()
    
    # Ensure all files have been saved before moving on
    S1_file_manager.shutdown()

    return FinalMask 
