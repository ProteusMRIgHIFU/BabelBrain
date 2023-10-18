'''
Processing of ZTE traditional (linear) conversion 
Miscouridou, M., Pineda-Pardo, J.A., Stagg, C.J., Treeby, B.E. and Stanziola, A., 
2022. Classical and learned MR to pseudo-CT mappings for accurate transcranial ultrasound simulation.
IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 69(10), pp.2896-2905.
'''
import nibabel
from nibabel import processing
from nibabel.spaces import vox2out_vox
import SimpleITK as sitk
import itk
import tempfile
import os
import scipy
from skimage.measure import label, regionprops
import numpy as np
from operator import itemgetter
import platform
from pathlib import Path
import sys
import subprocess
import shutil
from linetimer import CodeTimer



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

def RunElastix(reference,moving,finalname):
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    fixed_image = itk.imread(reference)
    moving_image = itk.imread(moving)
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
            parameter_object=parameter_object,
            log_to_console=True)
    itk.imwrite(result_image,finalname,compression=True)

def N4BiasCorrec(input,output=None,shrinkFactor=4,
                convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},):
    inputImage = sitk.ReadImage(input, sitk.sitkFloat32)
    image = inputImage

    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    if shrinkFactor > 1:
        image = sitk.Shrink(
            inputImage, [shrinkFactor] * inputImage.GetDimension()
        )
        maskImage = sitk.Shrink(
            maskImage, [shrinkFactor] * inputImage.GetDimension()
        )
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(
        convergence['iters']
    )
    corrector.SetConvergenceThreshold(convergence['tol'])
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

    if output is not None:
        sitk.WriteImage(corrected_image_full_resolution, output)

    return corrected_image_full_resolution

def CTCorreg(InputT1,InputCT,CoregCT_MRI=0, ResampleFunc=None, ResampleBackend='OpenCL'):
    # CoregCT_MRI =0, do not coregister, just load data
    # CoregCT_MRI =1 , coregister CT-->MRI space
    # CoregCT_MRI =2 , coregister MRI-->CT space
    #Bias correction
    if CoregCT_MRI==0:
        return nibabel.load(InputCT)
    else:
        T1fnameBiasCorrec =os.path.splitext(InputT1)[0] + '_BiasCorrec.nii.gz' 
        N4BiasCorrec(InputT1,T1fnameBiasCorrec)
        #coreg
        if CoregCT_MRI==1:
            #we first upsample the T1W to the same resolution as CT
            # Set up for Resample call
            in_img=nibabel.load(T1fnameBiasCorrec)

            voxel_sizes = nibabel.load(InputCT).header.get_zooms()
            cval=in_img.get_fdata().min()

            in_shape = in_img.shape
            n_dim = len(in_shape)
            if voxel_sizes is not None:
                voxel_sizes = np.asarray(voxel_sizes)
                if voxel_sizes.ndim == 0:  # Scalar
                    voxel_sizes = np.repeat(voxel_sizes, n_dim)
            # Allow 2D images by promoting to 3D.  We might want to see what a slice
            # looks like when resampled into world coordinates
            if n_dim < 3:  # Expand image to 3D, make voxel sizes match
                new_shape = in_shape + (1,) * (3 - n_dim)
                data = np.asanyarray(in_img.dataobj).reshape(new_shape)  # 2D data should be small
                in_img = nibabel.Nifti1Image(data, in_img.affine, in_img.header)
                if voxel_sizes is not None and len(voxel_sizes) == n_dim:
                    # Need to pad out voxel sizes to match new image dimensions
                    voxel_sizes = tuple(voxel_sizes) + (1,) * (3 - n_dim)
            out_vox_map = vox2out_vox((in_img.shape, in_img.affine), voxel_sizes)
            
            fixed_image = ResampleFunc(in_img,out_vox_map, GPUBackend=ResampleBackend)


            T1fname_CTRes=os.path.splitext(InputT1)[0] + '_BiasCorrec_CT_res.nii.gz'
            fixed_image.to_filename(T1fname_CTRes)
            CTInT1W=os.path.splitext(InputCT)[0] + '_InT1.nii.gz'

            RunElastix(T1fname_CTRes,InputCT,CTInT1W)
            
            return nibabel.load(CTInT1W)
        else:
            T1WinCT=os.path.splitext(InputT1)[0] + '_InCT.nii.gz'
            RunElastix(InputCT,T1fnameBiasCorrec,T1WinCT)
            return nibabel.load(InputCT)


def BiasCorrecAndCoreg(InputT1,InputZTE,img_mask):
    #Bias correction
    T1fnameBiasCorrec =os.path.splitext(InputT1)[0] + '_BiasCorrec.nii.gz'

    N4BiasCorrec(InputT1,T1fnameBiasCorrec)

    ZTEfnameBiasCorrec=os.path.splitext(InputZTE)[0] + '_BiasCorrec.nii.gz'

    N4BiasCorrec(InputZTE,ZTEfnameBiasCorrec)
    #coreg

    ZTEInT1W=os.path.splitext(InputZTE)[0] + '_InT1.nii.gz'
    RunElastix(T1fnameBiasCorrec,ZTEfnameBiasCorrec,ZTEInT1W)
    
    img=sitk.ReadImage(T1fnameBiasCorrec, sitk.sitkFloat32)
    try:
        img_out=img*sitk.Cast(img_mask,sitk.sitkFloat32)
    except:
        img_mask.SetSpacing(img.GetSpacing()) # some weird rounding can occur, so we try again
        img_out=img*sitk.Cast(img_mask,sitk.sitkFloat32)
    sitk.WriteImage(img_out, T1fnameBiasCorrec)

    img=sitk.ReadImage(ZTEInT1W, sitk.sitkFloat32)
    img_out = img*sitk.Cast(img_mask,sitk.sitkFloat32)
    sitk.WriteImage(img_out, ZTEInT1W)
    return T1fnameBiasCorrec,ZTEInT1W

def ConvertZTE_pCT(InputT1,InputZTE,TMaskItk,SimbsPath,ThresoldsZTEBone=[0.1,0.6],SimbNIBSType='charm'):
    print('converting ZTE to pCT with range',ThresoldsZTEBone)

    if SimbNIBSType=='charm':
        #while charm is much more powerful to segment skull regions, we need to calculate the meshes ourselves
        charminput = os.path.join(SimbsPath,'final_tissues.nii.gz')
        charm= nibabel.load(charminput)
        charmdata=np.ascontiguousarray(charm.get_fdata())[:,:,:,0]
        arrSkin=charmdata>0 #this mimics what the old headreco does for skin
        arrMask=(charmdata==1) | (charmdata==2) | (charmdata==3) | (charmdata==9) #this mimics what the old headreco does for csf
        label_img=label(charmdata==0)
        regions= regionprops(label_img)
        regions=sorted(regions,key=lambda d: d.area) #we eliminate the large background region
        arrCavities=(label_img!=0) &(label_img!=regions[-1].label)
    else:
        InputBrainMask=os.path.join(SimbsPath,'csf.nii.gz')
        SkinMask=os.path.join(SimbsPath,'skin.nii.gz')
        CavitiesMask=os.path.join(SimbsPath,'cavities.nii.gz')
        # Load T1 and ZTE
        volumeMask = nibabel.load(InputBrainMask)
        volumeSkin = nibabel.load(SkinMask)
        volumeCavities = nibabel.load(CavitiesMask)
        arrMask=volumeMask.get_fdata()
        arrSkin=volumeSkin.get_fdata()
        arrCavities=volumeCavities.get_fdata()
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        HeadMask=os.path.join(tmpdirname,'tissueregion.nii.gz')
        sitk.WriteImage(TMaskItk,HeadMask)
        
        volumeT1 = nibabel.load(InputT1)
        volumeZTE = nibabel.load(InputZTE)
        volumeHead = nibabel.load(HeadMask)
        
        arrZTE=volumeZTE.get_fdata()
        arrHead=volumeHead.get_fdata()
        
        
        
        maskedZTE =arrZTE.copy()
        maskedZTE[arrMask==0]=-1000
        
        cutoff=np.percentile(maskedZTE[maskedZTE>-500].flatten(),95)
        
        
        arrZTE/=cutoff
        
        arrZTE[arrHead==0]=-0.5

        arrGauss=arrZTE.copy()
        arrGauss[scipy.ndimage.binary_erosion(arrHead,iterations=3)==0]=np.max(arrGauss)
        arr=(arrGauss>=ThresoldsZTEBone[0]) & (arrGauss<=ThresoldsZTEBone[1])
        
        label_img = label(arr)
        def pixelcount(regionmask):
            return np.sum(regionmask)
        props = regionprops(label_img, extra_properties=(pixelcount,))
        props = sorted(props, key=itemgetter('pixelcount'), reverse=True)
        arr2=scipy.ndimage.binary_closing(label_img==props[0].label,structure=np.ones((11,11,11))).astype(np.uint8)

        
        arrCT=np.zeros_like(arrGauss)
        arrCT[arrSkin==0]=-1000 
        arrCT[arrSkin!=0]=42.0 #soft tissue
        arrCT[arr2!=0]=-2085*arrZTE[arr2!=0]+ 2329.0
        arrCT[arrCT<-1000]=-1000 #air
        arrCT[arrCT>3300]=-1000 #air 
        arrCT[arrCavities!=0]=-1000
        
        pCT = nibabel.Nifti1Image(arrCT,affine=volumeZTE.affine)
        CTfname=InputZTE.split('-InT1.nii.gz')[0]+'-pCT.nii.gz'
        nibabel.save(pCT,CTfname)
    return pCT
