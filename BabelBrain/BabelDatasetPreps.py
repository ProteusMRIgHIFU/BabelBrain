'''
Tools to generate files for acoustic/viscoleastic simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 23, 2021
     last update   - Sep 30, 2022

'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import os
import glob
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
import SimpleITK as sitk
import time
import gc
import yaml
from histoprint import text_hist, print_hist
import pandas as pd
import platform
import sys
from linetimer import CodeTimer
import re
try:
    import CTZTEProcessing
    from CTZTEProcessing import SaveHashInfo, GetBlake2sHash
except:
    from . import CTZTEProcessing
    from .CTZTEProcessing import SaveHashInfo, GetBlake2sHash
import tempfile

try:
    from ConvMatTransform import ReadTrajectoryBrainsight, GetIDTrajectoryBrainsight,read_itk_affine_transform,itk_to_BSight
except:
    from .ConvMatTransform import ReadTrajectoryBrainsight, GetIDTrajectoryBrainsight,read_itk_affine_transform,itk_to_BSight

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



if sys.platform in ['linux','win32']:
    print('importing cupy')
    import cupy 
    import cupyx 
    from cupyx.scipy import ndimage as cndimage

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
    else:
        MedianCOMPUTING_BACKEND='Metal'

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
    else:
        VoxelizeCOMPUTING_BACKEND='Metal'

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
    else:
        MapFilterCOMPUTING_BACKEND='Metal'

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
    else:
        ResampleFilterCOMPUTING_BACKEND='Metal'

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
    else:
        BinaryClosingFilterCOMPUTING_BACKEND='Metal'

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
    else:
        LabelImageCOMPUTING_BACKEND='Metal'

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

def DoIntersect(Mesh1,Mesh2):
    #We took now some extra steps for broken meshes
    Mesh1_intersect =trimesh.boolean.intersection((Mesh1,Mesh2),engine='blender')
    try:
        dummy = Mesh1_intersect.triangles #if empty, this trigger an error
    except:
        print('mesh is invalid... trying to fix')
        Mesh1_intersect=FixMesh(Mesh1)
        Mesh1_intersect =trimesh.boolean.intersection((Mesh1_intersect,Mesh2),engine='blender')
    return Mesh1_intersect

def FixMesh(inmesh):
    with tempfile.TemporaryDirectory() as tmpdirname:
        inmesh.export(tmpdirname+os.sep+'__in.stl')
        pymeshfix.clean_from_file(tmpdirname+os.sep+'__in.stl', tmpdirname+os.sep+'__out.stl')
        fixmesh=trimesh.load_mesh(tmpdirname+os.sep+'__out.stl')
        os.remove(tmpdirname+os.sep+'__in.stl')
        os.remove(tmpdirname+os.sep+'__out.stl')
    return fixmesh

def GenerateFileNames(SimbNIBSDir,SimbNIBSType,T1Source_nii,T1Conformal_nii,CT_or_ZTE_input,CTType,CoregCT_MRI,prefix):
    inputfiles = {}
    outputfiles = {}

    # T1W file name
    inputfiles['T1input'] = T1Source_nii

    # CT,ZTE, or PETRA file name if a file is given
    if CT_or_ZTE_input is not None:
        inputfiles['CTZTEinput'] = CT_or_ZTE_input

    # SimbNIBS input file name
    if SimbNIBSType == 'charm':
        inputfiles['SimbNIBSinput'] = SimbNIBSDir + os.sep + 'final_tissues.nii.gz'
        outputfiles['ReuseSimbNIBS'] = SimbNIBSDir + os.sep + 'final_tissues_babelbrain_reuse.npy' # This file is only used to confirm if inputs can be reused
    else:
        inputfiles['SimbNIBSinput'] = SimbNIBSDir + os.sep + 'skin.nii.gz'
        outputfiles['ReuseSimbNIBS'] = SimbNIBSDir + os.sep + 'skin_babelbrain_reuse.npy' # This file is only used to confirm if inputs can be reused

    # SimbNIBS stl file names
    outputfiles['Skull_STL'] = SimbNIBSDir+os.sep+'bone.stl'
    outputfiles['CSF_STL'] = SimbNIBSDir+os.sep+'csf.stl'
    outputfiles['Skin_STL'] = SimbNIBSDir+os.sep+'skin.stl'
    
    # BiasCorrec and Coreg File Names
    if CT_or_ZTE_input is not None:
        BaseNameInT1 = os.path.splitext(T1Conformal_nii)[0]
        if '.nii.gz' in T1Conformal_nii:
            BaseNameInT1 =os.path.splitext(BaseNameInT1)[0]

        if CTType in [2,3]: # ZTE/PETRA
            outputfiles['T1fnameBiasCorrec'] = BaseNameInT1 + '_BiasCorrec.nii.gz'
            BaseNameInZTE=os.path.splitext(CT_or_ZTE_input)[0]
            if '.nii.gz' in CT_or_ZTE_input:
                BaseNameInZTE=os.path.splitext(BaseNameInZTE)[0]
            
            outputfiles['ZTEfnameBiasCorrec'] = BaseNameInZTE + '_BiasCorrec.nii.gz'
            outputfiles['ZTEInT1W'] = BaseNameInZTE+'_InT1.nii.gz'
            outputfiles['pCTfname'] = outputfiles["ZTEfnameBiasCorrec"] + '_pCT.nii.gz'

        else: # CT
            if CoregCT_MRI==0:
                pass
            elif CoregCT_MRI == 1:
                outputfiles['T1fnameBiasCorrec'] = BaseNameInT1 + '_BiasCorrec.nii.gz'
                outputfiles['T1fname_CTRes'] = BaseNameInT1 + '_BiasCorrec_CT_res.nii.gz'

                CTInT1W = os.path.splitext(CT_or_ZTE_input)[0]
                if '.nii.gz' in CT_or_ZTE_input:
                    CTInT1W=os.path.splitext(CTInT1W)[0]
                
                CTInT1W += '_InT1.nii.gz'
                outputfiles['CTInT1W'] = CTInT1W
            else:
                outputfiles['T1WinCT'] = BaseNameInT1 + '_InCT.nii.gz'

        # Intermediate mask file name
        outputfiles['ReuseMask'] = os.path.dirname(T1Conformal_nii)+os.sep+prefix+'ReuseMask.nii.gz' # This file is only used to confirm if previously generated intermediate mask can be reused
        
        # CT file name
        outputfiles['CTfname'] = os.path.dirname(T1Conformal_nii)+os.sep+prefix+'CT.nii.gz'

    return inputfiles, outputfiles

def CheckReuseFiles(infnames, outfnames, currentCTType=None, currentHUT = None, currentZTER = None):
    # This function grabs checksum and parameter information stored in the header of intermediate nifti files 
    # generated in previous BabelBrain runs, and determines if those files can be reused to speed up current 
    # run.

    currentHashes = ""
    expectedHashes = []
    expectedCTTypes = []
    expectedHUThresholds = []
    expectedZTERanges = []
    prevfiles = {}
    CTTypePattern = "CTType=\w+"
    HUTPattern = "HUT=\d+"
    ZTERPattern = "ZTER=.+\)"

    if currentCTType is not None:
        if currentCTType == 1:
            currentCTType = "CT"
        elif currentCTType == 2:
            currentCTType = "ZTE"
        else: #3
            currentCTType = "PETRA"

    if currentHUT is not None:
        currentHUT = int(currentHUT)

    if currentZTER is not None:
        currentZTER = str(currentZTER)

    # Grab checksum and parameter information from existing nifti file headers
    for key,file in outfnames.items():
        prevfiles[key] = outfnames[key]

        # Check if files already exist
        if not os.path.isfile(file):

            # If file doesn't exist for current target, use existing file for a different target
            target = re.search("\w+(?=_[a-zA-Z0-9]+_[a-zA-Z0-9]+Hz.+)",file)

            if target is not None:
                prevTargetSearch = re.sub(target.group(),"*",file)
                prevTargets = glob.glob(prevTargetSearch)

                if len(prevTargets) > 0:
                    prevfiles[key] = prevTargets[0] # Use first equivalent file at different target 
                else:
                    print(f"Previous files use different transducer, frequency, or PPW")
                    return False, outfnames
            else:
                print(f"Previous files don't exist")
                return False, outfnames

        if ".nii.gz" in prevfiles[key]:
            # Load stored information
            try:
                prevfile = nibabel.load(prevfiles[key])
                description = str(prevfile.header['descrip'].astype('str'))
            except:
                print(f"{prevfiles[key]} was corrupted")
                continue
        elif "babelbrain_reuse.npy" in prevfiles[key]:
            # Load stored information
            try:
                prevfile = np.load(prevfiles[key])
                description = str(prevfile.astype('str'))
            except:
                print(f"final_tissues_babelbrain_reuse.npy was corrupted")
                continue
        else:
            continue # No information stored in stl, npz, etc files

        # Extract parameter information 
        CTType = re.search(CTTypePattern,description)
        HUT = re.search(HUTPattern,description)
        ZTER = re.search(ZTERPattern,description)

        if CTType is not None:
            expectedCTTypes += [CTType.group()]
            description = re.sub(CTTypePattern + ",", "",description)

        if HUT is not None:
            expectedHUThresholds += [HUT.group()]
            description = re.sub(HUTPattern + ",", "",description)

        if ZTER is not None:
            expectedZTERanges += [ZTER.group()]
            description = re.sub(ZTERPattern + ",", "",description)
        
        # Extract checksum information
        expectedHashes += [description]
    
    # Grab current checksum information for both input and output files
    for f in (infnames | prevfiles).values():
        currentHashes += GetBlake2sHash(f)+","

    # Check all expected hashes match current hashes
    for expectedHash in expectedHashes:
        if expectedHash not in currentHashes:
            print(f"Missing expected hash")
            return False, outfnames

    # Check all CT Type of previous files match current ones
    if currentCTType is not None and len(expectedCTTypes) == 0:
        print(f"Previous files didn't use CT")
        return False, outfnames
    else:
        for expectedCTType in expectedCTTypes:
            if expectedCTType != f"CTType={currentCTType}":
                print(f"Previous files used different CT Type")
                return False, outfnames
                
    # Check all HU Threshold of previous files match current ones
    if currentHUT is not None and len(expectedHUThresholds) == 0:
        print(f"Previous files didn't have HUThreshold value")
        return False, outfnames
    else:
        for expectedHUT in expectedHUThresholds:
            if int(expectedHUT[4:]) != currentHUT:
                print(f"Previous files used different HUThreshold")
                return False, outfnames
        
    # Check all ZTE Range of previous files match current ones
    if currentZTER is not None and len(expectedZTERanges) == 0:
        print(f"Previous files didn't have ZTE/PETRA Range values")
        return False, outfnames
    else:
        for expectedZTER in expectedZTERanges:
            if expectedZTER[5:] != currentZTER:
                print(f"Previous files used different ZTE/PETRA Range")
                return False, outfnames
    
    print("Reusing previously generated files")
    return True, prevfiles

#process first with SimbNIBS
def GetSkullMaskFromSimbNIBSSTL(SimbNIBSDir='4007/4007_keep/m2m_4007_keep/',
                                SimbNIBSType='charm',# indicate if processing was done with charm or headreco
                                T1Source_nii ='4007/4007_keep/m2m_4007_keep/T1.nii.gz',
                                T1Conformal_nii='4007/4007_keep/m2m_4007_keep/T1fs_conform.nii.gz', #be sure it is the conformal 
                                CT_or_ZTE_input=None,
                                CTType = 1,
                                CoregCT_MRI=0, #if using CT, 0 does not coreg (assuming this was done previously), 1 from CT to MRI
                                ZTERange=(0.1,0.6),
                                HUThreshold=300.0,
                                HUCapThreshold=2100.0,
                                CT_quantification=10, #bits
                                Mat4Trajectory=None, 
                                TrajectoryType='brainsight',                               
                                Foc=135.0, #Tx focal length
                                FocFOV=165.0, #Tx focal length used for FOV subvolume
                                TxDiam=157.0, # Tx aperture diameter used for FOV subvolume
                                Location=[27.5, -42, 42],#RAS location of target ,
                                TrabecularProportion=0.8, #proportion of trabecular bone
                                SpatialStep=1500/500e3/9*1e3, #step of mask to reconstruct , mm
                                prefix='', #Id to add to output file for identification
                                bDoNotAlign=False, #Use this to just move the Tx to match the coordinate with Tx facing S-->I, otherwise it will simulate the alignment of the Tx to be normal to the skull surface
                                nIterationsAlign=10, # number of iterations to align the tx, 10 is often way more than enough for shallow targets
                                InitialAligment='HF',
                                bPlot=True,
                                bAlignToSkin=False,
                                factorEnlargeRadius=1.05,
                                bApplyBOXFOV=False,
                                DeviceName=''): #created reduced FOV
    '''
    Generate masks for acoustic/viscoelastic simulations. 
    It creates an Nifti file that is in subject space using as main inputs the output files of the headreco tool and location of coordinates where focal point is desired
    
    
    ABOUT:
         author        - Samuel Pichardo
         date          - June 23, 2021
         last update   - Nov 27, 2021
    
    '''
    #load T1W
    SavePath= os.path.dirname(T1Conformal_nii)
    T1Conformal=nibabel.load(T1Conformal_nii)
    baseaffine=T1Conformal.affine.copy()
    print('baseaffine',baseaffine,"\n")
    #sanity test to verify we have an isotropic scan
    assert(np.allclose(np.array(T1Conformal.header.get_zooms()),np.ones(3),rtol=1e-3))
    
    baseaffine[0,0]*=SpatialStep
    baseaffine[1,1]*=SpatialStep
    baseaffine[2,2]*=SpatialStep

    inputfilenames, outputfilenames = GenerateFileNames(SimbNIBSDir,SimbNIBSType,T1Source_nii,T1Conformal_nii,CT_or_ZTE_input,CTType,CoregCT_MRI,prefix)
    
    skull_stl=outputfilenames['Skull_STL']
    csf_stl=outputfilenames['CSF_STL']
    skin_stl=outputfilenames['Skin_STL']
    
    with CodeTimer("Checking if previously generated files can be reused", unit="s"):
        if CT_or_ZTE_input is None:
            bReuseFiles, prevoutputfilenames = CheckReuseFiles(inputfilenames,outputfilenames)
        else:
            if CTType in [2,3]:
                bReuseFiles, prevoutputfilenames = CheckReuseFiles(inputfilenames,outputfilenames,currentCTType=CTType,currentHUT=HUThreshold, currentZTER=ZTERange)
            else:
                bReuseFiles, prevoutputfilenames = CheckReuseFiles(inputfilenames,outputfilenames,currentCTType=CTType,currentHUT=HUThreshold)
        print(f"bReuseFiles: {bReuseFiles}")

    if SimbNIBSType=='charm':
        if bReuseFiles:
            TMaskItk=sitk.ReadImage(inputfilenames['SimbNIBSinput'], sitk.sitkFloat32)>0 #we also kept an SITK Object
        else:
            #while charm is much more powerful to segment skull regions, we need to calculate the meshes ourselves
            charminput = inputfilenames['SimbNIBSinput']
            charm= nibabel.load(charminput)
            charmdata=np.ascontiguousarray(charm.get_fdata())[:,:,:,0]
            AllTissueRegion=charmdata>0 #this mimics what the old headreco does for skin
            TMaskItk=sitk.ReadImage(charminput, sitk.sitkFloat32)>0 #we also kept an SITK Object

            BoneRegion=(charmdata>0) & (charmdata!=5) #this mimics what the old headreco does for bone
            CSFRegion=(charmdata==1) | (charmdata==2) | (charmdata==3) | (charmdata==9) #this mimics what the old headreco does for skin
            with CodeTimer("charm surface recon",unit='s'):
                skin_mesh=MaskToStl(AllTissueRegion,charm.affine)
                csf_mesh=MaskToStl(CSFRegion,charm.affine)
                skull_mesh=MaskToStl(BoneRegion,charm.affine)
                skin_mesh.export(skin_stl)
                csf_mesh.export(csf_stl)
                skull_mesh.export(skull_stl)

            SaveHashInfo(inputfilenames.values(),outputfilenames['ReuseSimbNIBS'])
    else:
        TMaskItk=sitk.ReadImage(inputfilenames['SimbNIBSinput'], sitk.sitkFloat32)>0 #we also kept an SITK Object
        SaveHashInfo(inputfilenames.values(),outputfilenames['ReuseSimbNIBS'])

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
    Cone=creation.cone(RadCone,HeightCone,transform=TransformationCone.copy())

    TransformationCone[2,3]=Location[2]
    CumulativeTransform=TransformationCone.copy() 
    
    TransformationCone=np.eye(4)
    TransformationCone[0,3]=-Location[0]
    TransformationCone[1,3]=-Location[1]
    TransformationCone[2,3]=-Location[2]

    CumulativeTransform=TransformationCone@CumulativeTransform

    Cone.apply_transform(TransformationCone)
    
    
    RMat=RMat[:3,:3]
    
    TransformationCone=np.eye(4)
    TransformationCone[0,3]=Location[0]
    TransformationCone[1,3]=Location[1]
    TransformationCone[2,3]=Location[2]
    
    TransformationCone[0:3,0:3]=RMat

    Cone.apply_transform(TransformationCone)

    CumulativeTransform=TransformationCone@CumulativeTransform

    print('Final RMAT')
    print(RMat)
    
    skin_mesh = trimesh.load_mesh(skin_stl)
    #we intersect the skin region with a cone region oriented in the same direction as the acoustic beam
    skin_mesh =DoIntersect(skin_mesh,Cone)

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
    DimsBox=(np.max(AffIJK,axis=0)[:3]-np.min(AffIJK,axis=0)[:3]+1)*SpatialStep
    TransformationBox=np.eye(4)
    TransformationBox[2,3]=-DimsBox[2]/2
    TransformationBox[2,3]+=FocFOV-Foc
    
    BoxFOV=creation.box(DimsBox,
                        transform=TransformationBox)
    BoxFOV.apply_transform(CumulativeTransform)
    BoxFOV.export(os.path.dirname(T1Conformal_nii)+os.sep+prefix+'_box_FOV.stl')
      
    ##################### And we repeat and complete data extraction
    baseaffine=T1Conformal.affine.copy()
    baseaffine[0,0]*=SpatialStep
    baseaffine[1,1]*=SpatialStep
    baseaffine[2,2]*=SpatialStep


    skull_mesh = trimesh.load_mesh(skull_stl)
    csf_mesh = trimesh.load_mesh(csf_stl)
    skin_mesh = trimesh.load_mesh(skin_stl)

    if bApplyBOXFOV:
        skull_mesh=DoIntersect(skull_mesh,BoxFOV)
        csf_mesh  =DoIntersect(csf_mesh,  BoxFOV)
        skin_mesh =DoIntersect(skin_mesh, BoxFOV)
    
    #we first subtract to find the pure bone region
    if VoxelizeFilter is None:
        while(True):
            try:
                with CodeTimer("skull voxelization",unit='s'):
                    skull_grid = skull_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                with CodeTimer("brain voxelization",unit='s'):
                    csf_grid = csf_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                with CodeTimer("skin voxelization",unit='s'):
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
            fct = nibabel.load(prevoutputfilenames['ReuseMask'])

            # Load in appropriate rCT and resave fct under current file name
            if CTType in [2,3]:
                rCT = nibabel.load(outputfilenames['pCTfname'])

                SaveHashInfo([outputfilenames['pCTfname']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold, ZTER=ZTERange)
            else:
                if CoregCT_MRI == 0:
                    rCT = nibabel.load(inputfilenames['CTZTEinput'])

                    SaveHashInfo([outputfilenames['ReuseSimbNIBS'],outputfilenames['Skull_STL'],outputfilenames['CSF_STL'],outputfilenames['Skin_STL']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold)
                elif CoregCT_MRI == 1:
                    rCT = nibabel.load(outputfilenames['CTInT1W'])

                    SaveHashInfo([outputfilenames['CTInT1W']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold)
                else:
                    rCT = nibabel.load(outputfilenames['T1WinCT'])

                    SaveHashInfo([outputfilenames['T1WinCT']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold)

            rCTdata=rCT.get_fdata()
        else:
            if CTType in [2,3]:
                print('Processing ZTE/PETRA to pCT')
                bIsPetra = CTType==3
                with CodeTimer("Bias and coregistration ZTE/PETRA to T1",unit='s'):
                    rT1,rZTE=CTZTEProcessing.BiasCorrecAndCoreg(T1Conformal_nii,CT_or_ZTE_input,TMaskItk,outputfilenames)
                with CodeTimer("Conversion ZTE/PETRA to pCT",unit='s'):
                    rCT = CTZTEProcessing.ConvertZTE_PETRA_pCT(rT1,rZTE,TMaskItk,os.path.dirname(skull_stl),outputfilenames,
                        ThresoldsZTEBone=ZTERange,SimbNIBSType=SimbNIBSType,bIsPetra=bIsPetra)
            else:
                with CodeTimer("Coregistration CT to T1",unit='s'):
                    rCT=CTZTEProcessing.CTCorreg(T1Conformal_nii,CT_or_ZTE_input, outputfilenames, CoregCT_MRI, bReuseFiles,ResampleFilter, ResampleFilterCOMPUTING_BACKEND)
            rCTdata=rCT.get_fdata()
            hist = np.histogram(rCTdata[rCTdata>HUThreshold],bins=15)
            print('*'*40)
            print_hist(hist, title="CT HU", symbols=r"=",fg_colors="0",bg_colors="0",columns=80)
            rCTdata[rCTdata>HUCapThreshold]=HUCapThreshold
            sf=np.round((np.ones(3)*2)/rCT.header.get_zooms()).astype(int)
            sf2=np.round((np.ones(3)*5)/rCT.header.get_zooms()).astype(int)
            with CodeTimer("median filter CT",unit='s'):
                print('Theshold for bone',HUThreshold)
                if sys.platform in ['linux','win32']:
                    gfct=cupy.asarray((rCTdata>HUThreshold))
                    gfct=cndimage.median_filter(gfct,sf)
                else:
                    fct=ndimage.median_filter(rCTdata>HUThreshold,sf,mode='constant',cval=0)

            with CodeTimer("binary closing CT",unit='s'):
                if sys.platform in ['linux','win32']:
                    fct=gfct.get()
                fct = BinaryClosingFilter(fct, structure=np.ones(sf2,dtype=int), GPUBackend=BinaryClosingFilterCOMPUTING_BACKEND)
            fct=nibabel.Nifti1Image(fct.astype(np.float32), affine=rCT.affine)

            # fct can be reused in future sims to save time
            if CTType in [2,3]:
                SaveHashInfo([outputfilenames['pCTfname']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold,ZTER=ZTERange)
            else:
                if CoregCT_MRI == 0:
                    SaveHashInfo([outputfilenames['ReuseSimbNIBS'],outputfilenames['Skull_STL'],outputfilenames['CSF_STL'],outputfilenames['Skin_STL']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold)
                elif CoregCT_MRI == 1:
                    SaveHashInfo([outputfilenames['CTInT1W']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold)
                else:
                    SaveHashInfo([outputfilenames['T1WinCT']],outputfilenames['ReuseMask'],fct,CTType=CTType,HUT=HUThreshold)

        mask_nifti2 = nibabel.Nifti1Image(FinalMask, affine=baseaffineRot)

        with CodeTimer("median filter CT mask extrapol",unit='s'):
            if ResampleFilter is None:
                nfct=processing.resample_from_to(fct,mask_nifti2,mode='constant',cval=0)
            else:
                nfct = ResampleFilter(fct,mask_nifti2,mode='constant',cval=0,GPUBackend=ResampleFilterCOMPUTING_BACKEND)
        nfct=np.ascontiguousarray(nfct.get_fdata())>0.5

        ##We will create an smooth surface
        with CodeTimer("skull surface CT",unit='s'):
            if LabelImage is None:
                label_img=label(nfct)
            else:
                label_img = LabelImage(nfct, GPUBackend=LabelImageCOMPUTING_BACKEND)
            regions= regionprops(label_img)
            regions=sorted(regions,key=lambda d: d.area)
            nfct=label_img==regions[-1].label
            smct=MaskToStl(nfct,baseaffineRot)
                
        with CodeTimer("CT skull voxelization",unit='s'):
            if VoxelizeFilter is None:
                ct_grid = smct.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
            else:
                ct_grid=VoxelizeFilter(smct,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)
        
        XYZ=ct_grid
        XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1),dtype=ct_grid.dtype))).T
        AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(int).T

        nfct=np.zeros_like(FinalMask)

        inds=(AffIJK[:,0]<nfct.shape[0])&\
            (AffIJK[:,1]<nfct.shape[1])&\
            (AffIJK[:,2]<nfct.shape[2])
        AffIJK=AffIJK[inds,:]

        nfct[AffIJK[:,0],AffIJK[:,1],AffIJK[:,2]]=1

        with CodeTimer("CT median filter",unit='s'):
            if sys.platform in ['linux','win32']:
                gnfct=cupy.asarray(nfct.astype(np.uint8))
                gnfct=cndimage.median_filter(gnfct,7)
                nfct=gnfct.get()
            else:
                if MedianFilter is None:
                    nfct=ndimage.median_filter(nfct.astype(np.uint8),7)
                else:
                    nfct=MedianFilter(nfct.astype(np.uint8),GPUBackend=MedianCOMPUTING_BACKEND)
            nfct=nfct!=0

        del XYZ
        del ct_grid


        ############
        with CodeTimer("CT extrapol",unit='s'):
            print('rCTdata range',rCTdata.min(),rCTdata.max())
            rCT = nibabel.Nifti1Image(rCTdata, rCT.affine, rCT.header)
            if ResampleFilter is None:
                nCT=processing.resample_from_to(rCT,mask_nifti2,mode='constant',cval=rCTdata.min())
            else:
                nCT=ResampleFilter(rCT,mask_nifti2,mode='constant',cval=rCTdata.min(),GPUBackend=ResampleFilterCOMPUTING_BACKEND)
            ndataCT=np.ascontiguousarray(nCT.get_fdata()).astype(np.float32)
            ndataCT[ndataCT>HUCapThreshold]=HUCapThreshold
            print('ndataCT range',ndataCT.min(),ndataCT.max())
            ndataCT[nfct==False]=0

        with CodeTimer("CT binary_dilation",unit='s'):
            BinMaskConformalCSFRot= ndimage.binary_dilation(BinMaskConformalCSFRot,iterations=6)
        with CodeTimer("FinalMask[BinMaskConformalCSFRot]=4",unit='s'):
            FinalMask[BinMaskConformalCSFRot]=4  
            FinalMask[BinMaskConformalSkullRot==1]=4
        #brain
        with CodeTimer("FinalMask[nfct]=2",unit='s'):
            FinalMask[nfct]=2  #bone
        #we do a cleanup of islands 
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
            AllLabels.append(l.label)

        FinalMask[np.isin(label_img,np.array(AllLabels))]=4
            
        CTBone=ndataCT[nfct]
        CTBone[CTBone<HUThreshold]=HUThreshold #we cut off to avoid problems in acoustic sim
        ndataCT[nfct]=CTBone
        maxData=ndataCT[nfct].max()
        minData=ndataCT[nfct].min()
        
        A=maxData-minData
        M = 2**CT_quantification-1
        ResStep=A/M 
        qx = ResStep *  np.round( (M/A) * (ndataCT[nfct]-minData) )+ minData
        ndataCT[nfct]=qx
        UniqueHU=np.unique(ndataCT[nfct])
        print('Unique CT values',len(UniqueHU))
        CTCalfname = os.path.dirname(T1Conformal_nii)+os.sep+prefix+'CT-cal.npz'
        np.savez_compressed(CTCalfname,UniqueHU=UniqueHU)

        with CodeTimer("Mapping unique values",unit='s'):
            if MapFilter is None:
                ndataCTMap=np.zeros(ndataCT.shape,np.uint32)
                for n,d in enumerate(UniqueHU):
                    ndataCTMap[ndataCT==d]=n
                ndataCTMap[nfct==False]=0
            else:
                ndataCTMap=MapFilter(ndataCT,nfct.astype(np.uint8),UniqueHU,GPUBackend=MapFilterCOMPUTING_BACKEND)

            smct.export(os.path.dirname(T1Conformal_nii)+os.sep+prefix+'CT_smooth.stl')
            nCT=nibabel.Nifti1Image(ndataCTMap, nCT.affine, nCT.header)

            if CTType in [2,3]:
                SaveHashInfo([outputfilenames['ReuseMask']],outputfilenames['CTfname'],nCT,CTType=CTType,HUT=HUThreshold, ZTER=ZTERange)
            else:
                SaveHashInfo([outputfilenames['ReuseMask']],outputfilenames['CTfname'],nCT,CTType=CTType,HUT=HUThreshold)

    with CodeTimer("final median filter ",unit='s'):
        if sys.platform in ['linux','win32']:
            gFinalMask=cupy.asarray(FinalMask.astype(np.uint8))
            gFinalMask=cndimage.median_filter(gFinalMask,7)
            FinalMask=gFinalMask.get()
        else:
            if MedianFilter is None:
                FinalMask=ndimage.median_filter(FinalMask.astype(np.uint8),7)
            else:
                FinalMask=MedianFilter(FinalMask.astype(np.uint8),GPUBackend=MedianCOMPUTING_BACKEND)
    
    #we extract back the bone part
    BinMaskConformalSkullRot=FinalMask==2
    LineViewBone=BinMaskConformalSkullRot[LocFocalPoint[0],LocFocalPoint[1],LocFocalPoint[2]:]
    #we erode to establish the section of bone associated with trabecular
    DegreeErosion=int((LineViewBone.sum()*(1-TrabecularProportion)))
    print('DegreeErosion',DegreeErosion)
    Trabecula=ndimage.binary_erosion(BinMaskConformalSkullRot,iterations=DegreeErosion)

    FinalMask[Trabecula]=3 #trabecula
    
    FinalMask[LocFocalPoint[0],LocFocalPoint[1],LocFocalPoint[2]]=5 #focal point location
    mask_nifti2 = nibabel.Nifti1Image(FinalMask, affine=baseaffineRot)

    outname=os.path.dirname(T1Conformal_nii)+os.sep+prefix+'BabelViscoInput.nii.gz'
    mask_nifti2.to_filename(outname)

    with CodeTimer("resampling T1 to mask",unit='s'):
        if ResampleFilter is None:
            T1Conformal=processing.resample_from_to(T1Conformal,mask_nifti2,mode='constant',order=0,cval=T1Conformal.get_fdata().min())
        else:
            T1Conformal=ResampleFilter(T1Conformal,mask_nifti2,mode='constant',order=0,cval=T1Conformal.get_fdata().min(),GPUBackend=ResampleFilterCOMPUTING_BACKEND)
        T1W_resampled_fname=os.path.dirname(T1Conformal_nii)+os.sep+prefix+'T1W_Resampled.nii.gz'
        T1Conformal.to_filename(T1W_resampled_fname)
    
    if bPlot:
        plt.figure()
        plt.imshow(FinalMask[:,LocFocalPoint[1],:],cmap=plt.cm.jet)
        plt.gca().set_aspect(1.0)
        plt.colorbar();
    
    return FinalMask 
