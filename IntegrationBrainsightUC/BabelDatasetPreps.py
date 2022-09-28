'''
Tools to generate files for acoustic/viscoleastic simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 23, 2021
     last update   - Nov 27, 2021

'''
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import nibabel
from nibabel import processing
from scipy import ndimage
from trimesh import creation 
from scipy.spatial.transform import Rotation as R
import time
import gc
import yaml

import pandas as pd

def GetIDTrajectoryBrainsight(fname):
    names=['Target name', 
      'Loc. X','Loc. Y','Loc. Z',
      'm0n0','m0n1','m0n2',
      'm1n0','m1n1','m1n2',
      'm2n0','m2n1','m2n2']
    df=pd.read_csv(fname,comment='#',sep='\t',header=None,names=names,engine='python',usecols=names).iloc[0]  
    return df['Target name']

def ReadTrajectoryBrainsight(fname):
    names=['Target name', 
      'Loc. X','Loc. Y','Loc. Z',
      'm0n0','m0n1','m0n2',
      'm1n0','m1n1','m1n2',
      'm2n0','m2n1','m2n2']
    df=pd.read_csv(fname,comment='#',sep='\t',header=None,names=names,engine='python',usecols=names[1:]).iloc[0].to_numpy()
    Mat4=np.eye(4)
    Mat4[:3,3]=df[:3]
    Mat4[:3,0]=df[3:6]
    Mat4[:3,1]=df[6:9]
    Mat4[:3,2]=df[9:]
    
    return Mat4
from sys import platform
if platform in ['linux','win32']:
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
    if COMPUTING_BACKEND==2:
        MedianCOMPUTING_BACKEND='OpenCL'
    else:
        MedianCOMPUTING_BACKEND='Metal'
VoxelizeFilter=None
VoxelizeCOMPUTING_BACKEND=''

def InitVoxelizeGPUCallback(Callback=None,COMPUTING_BACKEND=2):
    global VoxelizeFilter
    global VoxelizeCOMPUTING_BACKEND
    VoxelizeFilter = Callback
    if COMPUTING_BACKEND==2:
        VoxelizeCOMPUTING_BACKEND='OpenCL'
    else:
        VoxelizeCOMPUTING_BACKEND='Metal'

def ConvertMNItoSubjectSpace(M1_C,DataPath,T1Conformal_nii,bUseFlirt=True,PathSimnNIBS=''):
    '''
    Convert MNI coordinates to patient coordinates using SimbNIBS converted data
    
    M1_C is a [3] list or array of 3 MNI coordinates
    DataPath is the path to the main subject data directory where the XXXX_T1fs_conform.nii.gz SimbNIBS T1W output file is located
    T1Conformal_nii is the name of SimbNIBS T1W output file, such as SimbNIBS_LIFU_01_T1fs_conform.nii.gz
    
    bUseFlirt (default True) indicates if usiing FSL flirt as the tool to convert from MNI to subject coordinates, otherwise use 
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

#process first with SimbNIBS
def GetSkullMaskFromSimbNIBSSTL(skull_stl='4007/4007_keep/m2m_4007_keep/bone.stl',
                                csf_stl='4007/4007_keep/m2m_4007_keep/csf.stl',
                                skin_stl='4007/4007_keep/m2m_4007_keep/skin.stl',
                                T1Conformal_nii='4007/4007_keep/m2m_4007_keep/T1fs_conform.nii.gz', #be sure it is the conformal 
                                CT_input=None,
                                CT_quantification=10, #bits
                                Mat4Brainsight=None,                                
                                Foc=135.0, #Tx focal length
                                FocFOV=165.0, #Tx focal length used for FOV subvolume
                                TxDiam=157.0, # Tx aperture diameter
                                Location=[27.5, -42, 42],#RAS location of target ,
                                TrabecularProportion=0.8, #proportion of trabecular bone
                                SpatialStep=1500/500e3/9*1e3, #step of mask to reconstruct , mm
                                prefix='', #Id to add to output file for identification
                                bDoNotAlign=False, #Use this to just move the Tx to match the coordinate with Tx facing S-->I, otherwise it will simulate the aligment of the Tx to be normal to the skull surface
                                nIterationsAlign=10, # number of iterations to align the tx, 10 is often way more than enough for shallow targets
                                InitialAligment='HF',
                                bPlot=True,
                                bAlignToSkin=False,
                                factorEnlargeRadius=1.1,
                                bApplyBOXFOV=False): #created reduced FOV
    '''
    Generate masks for acoustic/viscoelastic simulations. 
    It creates an Nifti file that is in subject space using as main inputs the output files of the headreco tool and location of coordinates where focal point is desired
    

    ABOUT:
         author        - Samuel Pichardo
         date          - June 23, 2021
         last update   - Nov 27, 2021
    
    '''
    #load T1W
    T1Conformal=nibabel.load(T1Conformal_nii)
    baseaffine=T1Conformal.affine.copy()
    print('baseaffine',baseaffine)
    #sanity test to verify we have an isotropic scan
    assert(np.allclose(np.array(T1Conformal.header.get_zooms()),np.ones(3),rtol=1e-3))
    
    baseaffine[0,0]*=SpatialStep
    baseaffine[1,1]*=SpatialStep
    baseaffine[2,2]*=SpatialStep



    #building a cone object representing acoustic beam pointing to desired location
    RadCone=TxDiam/2*factorEnlargeRadius
    if type(Foc) is tuple:
        Foc=Foc[0]
    HeightCone=np.sqrt(Foc**2-RadCone**2)
    print('HeightCone',HeightCone)
    
    InVAffine=np.linalg.inv(baseaffine)
    
    if Mat4Brainsight is None:    
        print('*'*40+'\n calculating optimal orientation\n'+'*'*40)

        if InitialAligment == 'HF':
            TransformationCone=np.eye(4)
            TransformationCone[2,2]=-1
            OrientVec=np.array([0,0,1]).reshape((1,3))
            TransformationCone[0,3]=Location[0]
            TransformationCone[1,3]=Location[1]
            TransformationCone[2,3]=Location[2]+HeightCone
        elif InitialAligment =='RL':
            TransformationCone=np.zeros((4,4))
            TransformationCone[0,2]=-1
            TransformationCone[1,1]=1
            TransformationCone[2,0]=1
            TransformationCone[3,3]=1
            OrientVec=np.array([1,0,0]).reshape((1,3))
            TransformationCone[0,3]=Location[0]+HeightCone
            TransformationCone[1,3]=Location[1]
            TransformationCone[2,3]=Location[2]
        elif InitialAligment == 'LR':
            TransformationCone=np.zeros((4,4))
            TransformationCone[0,2]=1
            TransformationCone[1,1]=1
            TransformationCone[2,0]=-1
            TransformationCone[3,3]=1
            OrientVec=np.array([-1,0,0]).reshape((1,3))
            TransformationCone[0,3]=Location[0]-HeightCone
            TransformationCone[1,3]=Location[1]
            TransformationCone[2,3]=Location[2]
        else:
            raise ValueError('InitialAligment not supported ', InitialAligment)
            
        Cone=creation.cone(RadCone,HeightCone,transform=TransformationCone.copy())

        if bAlignToSkin:
            reference_mesh = trimesh.load_mesh(skin_stl)
        else:
            reference_mesh = trimesh.load_mesh(skull_stl)
        
        #we iterate a few times calculating the normal vectors of the regions being crossed 
        #through the skull to ensure maximizing normal incidence
        #the cone is initially pointing superior->inferior (note that in trimesh the cone is oriented from the tip,
        #so in practice we use rather a [0,0,1] vector), 
        # by iterating a few times we can rotate the cone until the beam have an adequate normal
        #incident crossing through the skull. This could be improved using a minimization criteeria
        TransformationCone[2,3]=Location[2]
        CumulativeTransform=TransformationCone.copy()
        Cone.export(os.path.dirname(T1Conformal_nii)+os.sep+prefix+'_Base_cone.stl')

        if bDoNotAlign==False:
            for n in range(nIterationsAlign):
                #intersection of cone and skull surface
                reference_mesh_p1 =trimesh.boolean.intersection((reference_mesh,Cone),engine='blender')
                normals=reference_mesh_p1.face_normals #we recover normal vectors of intersected surface


                DNorm=np.dot(normals,OrientVec.T)
                SelNorm=DNorm>0.0 #we ignore any face pointing downwards
                print('SelNorm',SelNorm.sum(),DNorm.shape,DNorm[SelNorm].mean(),np.std(DNorm[SelNorm]))
                normals=normals[SelNorm.flatten(),:]
                AvgNormal=normals.mean(axis=0)
                AvgNormal=AvgNormal/np.linalg.norm(AvgNormal)
                print('avgNormal',AvgNormal)

                #we move the tip of the cone back to the isocenter first
                TransformationCone=np.eye(4)
                TransformationCone[0,3]=-Location[0]
                TransformationCone[1,3]=-Location[1]
                TransformationCone[2,3]=-Location[2]
                
                CumulativeTransform=TransformationCone@CumulativeTransform

                Cone.apply_transform(TransformationCone)
                
                #now we prepare a new transformation matrix that will rotate towards the
                #average normal vector of the surface of the skull being crossed and 
                #translated back to the intended location
                TransformationCone=np.eye(4)
                TransformationCone[0,3]=Location[0]
                TransformationCone[1,3]=Location[1]
                TransformationCone[2,3]=Location[2]
                
                #this calculates the transformation matrix to go from one vector to another
                RMat=R.align_vectors(AvgNormal.reshape((1,3)),OrientVec)[0].as_matrix()
                print('RMat',RMat,AvgNormal,OrientVec)

                OrientVec=AvgNormal.reshape((1,3))

                TransformationCone[0:3,0:3]=RMat
                
                Cone.apply_transform(TransformationCone)
                
                CumulativeTransform=TransformationCone@CumulativeTransform
            #we calculate the final transformation matrix that rotates from the S->I direction to the direction 
            # that ensures normal incidence
            RMat=R.align_vectors(AvgNormal.reshape((1,3)),[[0,0,1]])[0].as_matrix()
        else:
            RMat=np.eye(3)
        print('DONE with iterations')
    else:
        print('*'*40+'\n Reading orientation and target location directly from Brainsight export\n'+'*'*40)
        RMat=ReadTrajectoryBrainsight(Mat4Brainsight)
        print('Brainsight Matrix\n',RMat)
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

#        TransformationCone[2,3]=Location[2]
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
    
    #we save the final cone profile
    Cone.export(os.path.dirname(T1Conformal_nii)+os.sep+prefix+'_cone.stl')
      
    skin_mesh = trimesh.load_mesh(skin_stl)
    #we intersect the skin region with a cone region oriented in the same direction as the acoustic beam
    skin_mesh =trimesh.boolean.intersection((skin_mesh,Cone),engine='blender')
    print('doing intersection voxelization...')
    t0=time.time()
    #we obtain the list of Cartesian voxels inside the skin region intersected by the cone    

    if VoxelizeFilter is None:  
        skin_grid = skin_mesh.voxelized(SpatialStep,max_iter=30).fill().points
    else:
        skin_grid = VoxelizeFilter(skin_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)

    print('time to voxelize',time.time()-t0)
    
    x_vec=np.arange(skin_grid[:,0].min(),skin_grid[:,0].max()+SpatialStep,SpatialStep)
    y_vec=np.arange(skin_grid[:,1].min(),skin_grid[:,1].max()+SpatialStep,SpatialStep)
    z_vec=np.arange(skin_grid[:,2].min(),skin_grid[:,2].max()+SpatialStep,SpatialStep)
    
    Corner1=np.array([x_vec[0],y_vec[0],z_vec[0],1]).reshape((4,1))
    Corner2=np.array([x_vec[-1],y_vec[-1],z_vec[-1],1]).reshape((4,1))

    #we will produce one dataset (used only for sanity tests)
    # with the same orientation as the T1 scan and just enclosing the list of points intersected
    AffIJKCorners=np.floor(np.dot(InVAffine,np.hstack((Corner1,Corner2)))).astype(np.int).T
    AffIJKCornersMin=np.min(AffIJKCorners,axis=0).reshape((4,1))
    NewOrig=np.dot(baseaffine,AffIJKCornersMin)
    
    baseaffine[:,3]=NewOrig.flatten()
    
    
    print('baseaffine',baseaffine)
    baseaffineRot=baseaffine.copy()
    RMat4=np.eye(4)
    RMat4[:3,:3]=RMat*SpatialStep
    print('RMat4',RMat4)
    
    #now we prepare a rotated afffine matrix that will be aligned perpendicular to the cone
    tempnifti = nibabel.as_closest_canonical(T1Conformal)
    
    baseaffineRot=RMat4@tempnifti.affine
    
    print('baseaffineRot',baseaffineRot)
    
    InVAffine=np.linalg.inv(baseaffine)
    InVAffineRot=np.linalg.inv(baseaffineRot)

    XYZ=skin_grid
    #we make it a Nx4 array
    XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1)))).T
    
    #now we prepare the new dataset that is perpendicular to the cone direction
    #first we calculate the indexes i,j,k
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int).T
    NewOrig=baseaffineRot @np.array([AffIJK[:,0].min(),AffIJK[:,1].min(),AffIJK[:,2].min(),1]).reshape((4,1))
    baseaffineRot[:,3]=NewOrig.flatten()
    InVAffineRot=np.linalg.inv(baseaffineRot)
    
    ALoc=np.ones((4,1))
    ALoc[:3,0]=np.array(Location)
    XYZ=np.hstack((XYZ,ALoc))
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int).T
    
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
        skull_mesh=trimesh.boolean.intersection((skull_mesh,BoxFOV),engine='blender')
        csf_mesh  =trimesh.boolean.intersection((csf_mesh,  BoxFOV),engine='blender')
        skin_mesh =trimesh.boolean.intersection((skin_mesh, BoxFOV),engine='blender')
    
    #we first substract to find the pure bone region
    if VoxelizeFilter is None:
        while(True):
            try:
                print('doing skull voxelization...')
                t0=time.time()
                skull_grid = skull_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                print('time to voxelize',time.time()-t0)
                print('doing csf voxelization...')
                t0=time.time()
                csf_grid = csf_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                print('time to voxelize',time.time()-t0)
                print('doing skin voxelization...')
                t0=time.time()
                skin_grid = skin_mesh.voxelized(SpatialStep*0.75,max_iter=30).fill().points.astype(np.float32)
                print('time to voxelize',time.time()-t0)
                break
            except AttributeError as err:
                print("Repeating CSG boolean since once in while it returns an scene instead of a mesh....")
                print(err)
            else:
                raise err
    else:
        t0=time.time()
        skull_grid = VoxelizeFilter(skull_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)
        print('time to voxelize',time.time()-t0)
        print('doing csf voxelization...')
        t0=time.time()
        csf_grid = VoxelizeFilter(csf_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)
        print('time to voxelize',time.time()-t0)
        print('doing skin voxelization...')
        t0=time.time()
        skin_grid = VoxelizeFilter(skin_mesh,targetResolution=SpatialStep*0.75,GPUBackend=VoxelizeCOMPUTING_BACKEND)
        print('time to voxelize',time.time()-t0)
    
    #we obtain the list of Cartesian voxels in the whole skin region intersected by the cone    
        
    x_vec=np.arange(skin_grid[:,0].min(),skin_grid[:,0].max()+SpatialStep,SpatialStep)
    y_vec=np.arange(skin_grid[:,1].min(),skin_grid[:,1].max()+SpatialStep,SpatialStep)
    z_vec=np.arange(skin_grid[:,2].min(),skin_grid[:,2].max()+SpatialStep,SpatialStep)
    
    Corner1=np.array([x_vec[0],y_vec[0],z_vec[0],1]).reshape((4,1))
    Corner2=np.array([x_vec[-1],y_vec[-1],z_vec[-1],1]).reshape((4,1))
    
    #we will produce one dataset (used only for sanity tests)
    # with the same orientation as the T1 scan and just enclosing the list of points intersected
    AffIJKCorners=np.floor(np.dot(InVAffine,np.hstack((Corner1,Corner2)))).astype(np.int).T
    AffIJKCornersMin=np.min(AffIJKCorners,axis=0).reshape((4,1))
    NewOrig=np.dot(baseaffine,AffIJKCornersMin)
    
    baseaffine[:,3]=NewOrig.flatten()
    
    
    print('baseaffine',baseaffine)
    baseaffineRot=baseaffine.copy()
    RMat4=np.eye(4)
    RMat4[:3,:3]=RMat*SpatialStep
    print('RMat4',RMat4)
    
    #now we prepare a rotated afffine matrix that will be aligned perpendicular to the cone
    tempnifti = nibabel.as_closest_canonical(T1Conformal)
    
    baseaffineRot=RMat4@tempnifti.affine
    
    print('baseaffineRot',baseaffineRot)
    
    InVAffine=np.linalg.inv(baseaffine)
    InVAffineRot=np.linalg.inv(baseaffineRot)
    
    t0=time.time()
    XYZ=skin_grid
    XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1),dtype=skin_grid.dtype))).T
    #now we prepare the new dataset that is perpendicular to the cone direction
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int).T
    NewOrig=baseaffineRot @np.array([AffIJK[:,0].min(),AffIJK[:,1].min(),AffIJK[:,2].min(),1]).reshape((4,1))
    baseaffineRot[:,3]=NewOrig.flatten()
    InVAffineRot=np.linalg.inv(baseaffineRot)
    InVAffineRot=InVAffineRot.astype(skin_grid.dtype)
    del AffIJK
    ALoc=np.ones((4,1),dtype=skin_grid.dtype)
    ALoc[:3,0]=np.array(Location,dtype=skin_grid.dtype)
    XYZ=np.hstack((XYZ,ALoc))
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int).T
    
    LocFocalPoint=AffIJK[-1,:3]
    
    BinMaskConformalSkinRot=np.zeros(np.max(AffIJK,axis=0)[:3]+1)
    BinMaskConformalSkinRot[AffIJK[:,0],AffIJK[:,1],AffIJK[:,2]]=1.0
    
    print('time skin masking',time.time()-t0)
    del XYZ
    del skin_grid
    gc.collect()

    t0=time.time()

    XYZ=skull_grid
    XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1),dtype=skull_grid.dtype))).T
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int).T
    BinMaskConformalSkullRot=np.zeros_like(BinMaskConformalSkinRot)
    BinMaskConformalSkullRot[AffIJK[:,0],AffIJK[:,1],AffIJK[:,2]]=1.0
    print('time skull masking',time.time()-t0)
    del AffIJK
    del XYZ
    del skull_grid
    gc.collect()

    t0=time.time()

    XYZ=csf_grid
    XYZ=np.hstack((XYZ,np.ones((XYZ.shape[0],1),dtype=csf_grid.dtype))).T
    AffIJK=np.round(np.dot(InVAffineRot,XYZ)).astype(np.int).T
    BinMaskConformalCSFRot=np.zeros_like(BinMaskConformalSkinRot)
    BinMaskConformalCSFRot[AffIJK[:,0],AffIJK[:,1],AffIJK[:,2]]=1.0
    print('time csf masking',time.time()-t0)
    del AffIJK
    del XYZ
    del csf_grid
    gc.collect()
    
    FinalMask=BinMaskConformalSkinRot
    FinalMask[BinMaskConformalSkullRot==1]=2 #cortical
    FinalMask[BinMaskConformalCSFRot==1]=4#brain
    
    print('doing median_filter ...')
    t0=time.time()
    if platform in ['linux','win32']:
        gFinalMask=cupy.asarray(FinalMask.astype(np.uint8))
        gFinalMask=cndimage.median_filter(gFinalMask,7)
        FinalMask=gFinalMask.get()
    else:
        if MedianFilter is None:
            FinalMask=ndimage.median_filter(FinalMask.astype(np.uint8),7)
        else:
            FinalMask=MedianFilter(FinalMask.astype(np.uint8),GPUBackend=MedianCOMPUTING_BACKEND)
    print('time to filter',time.time()-t0)
    
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
    
    if CT_input is not None:
        rCT=nibabel.load(CT_input)
        nCT=processing.resample_from_to(rCT,mask_nifti2,mode='constant',cval=rCT.get_fdata().min())
        dataCT=np.ascontiguousarray(nCT.get_fdata()).astype(np.float32)
        BinMaskConformalSkullRot=np.ascontiguousarray(BinMaskConformalSkullRot)
        dataCT[BinMaskConformalSkullRot==False]=0
        CTBone=dataCT[BinMaskConformalSkullRot]
        CTBone[CTBone<0]=0 #we cut off to avoid problems in acoustic sim
        dataCT[BinMaskConformalSkullRot]=CTBone
        maxData=dataCT[BinMaskConformalSkullRot].max()
        minData=dataCT[BinMaskConformalSkullRot].min()

        A=maxData-minData
        M = 2**CT_quantification-1
        ResStep=A/M 
        qx = ResStep *  np.round( (M/A) * (dataCT[BinMaskConformalSkullRot]-minData) )+ minData
        dataCT[BinMaskConformalSkullRot]=qx
        UniqueHU=np.unique(dataCT[BinMaskConformalSkullRot])
        print('Unique CT values',len(UniqueHU))
        np.savez_compressed(os.path.dirname(T1Conformal_nii)+os.sep+prefix+'CT-cal',UniqueHU=UniqueHU)
        if MedianFilter is None:
            dataCTMap=np.zeros(dataCT.shape,np.uint32)
            for n,d in enumerate(UniqueHU):
                dataCTMap[dataCT==d]=n
            dataCTMap[BinMaskConformalSkullRot==False]=0
        else:
            dataCTMap=MapFilter(dataCT,BinMaskConformalSkullRot.astype(np.uint8),UniqueHU)

        nCT=nibabel.Nifti1Image(dataCTMap, nCT.affine, nCT.header)
        outname=os.path.dirname(T1Conformal_nii)+os.sep+prefix+'CT.nii.gz'
        nCT.to_filename(outname)
    if bPlot:
        plt.figure()
        plt.imshow(FinalMask[:,LocFocalPoint[1],:],cmap=plt.cm.jet)
        plt.gca().set_aspect(1.0)
        plt.colorbar();
    
    return FinalMask 
