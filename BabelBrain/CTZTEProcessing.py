'''
Processing of ZTE traditional (linear) conversion 
Miscouridou, M., Pineda-Pardo, J.A., Stagg, C.J., Treeby, B.E. and Stanziola, A., 
2022. Classical and learned MR to pseudo-CT mappings for accurate transcranial ultrasound simulation.
IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 69(10), pp.2896-2905.
'''
import ants
import nibabel
from nibabel import processing
#import itk
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
    path_script = os.path.join(resource_path(),"ExternalBin/elastix/run_mac.sh")
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = subprocess.run(
                ["zsh",
                path_script,
                reference,
                moving,
                tmpdirname], capture_output=True, text=True
        )
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        if result.returncode == 0:
            shutil.move(os.path.join(tmpdirname,'result.0.nii.gz'),finalname)

    if result.returncode != 0:
        raise SystemError("Error when trying to run elastix")

def CTCorreg(InputT1,InputCT,CoregCT_MRI=0):
    # CoregCT_MRI =0, do not coregister, just load data
    # CoregCT_MRI =1 , coregister CT-->MRI space
    # CoregCT_MRI =2 , coregister MRI-->CT space
    #Bias correction
    if CoregCT_MRI==0:
        return nibabel.load(InputCT)
    else:
        img = ants.image_read(InputT1)
        img_n4 = ants.n4_bias_field_correction(img)
        T1fnameBiasCorrec =os.path.splitext(InputT1)[0] + '_BiasCorrec.nii.gz'
        ants.image_write(img_n4,(T1fnameBiasCorrec))   

        #coreg
        if CoregCT_MRI==1:
            #we first upsample the T1W to the same resolution as CT

            fixed_image=nibabel.load(T1fnameBiasCorrec)
            fixed_image=processing.resample_to_output(fixed_image,voxel_sizes=nibabel.load(InputCT).header.get_zooms(),cval=fixed_image.get_fdata().min())
            T1fname_CTRes=os.path.splitext(InputT1)[0] + '_BiasCorrec_CT_res.nii.gz'
            fixed_image.to_filename(T1fname_CTRes)
            CTInT1W=os.path.splitext(InputCT)[0] + '_InT1.nii.gz'

            RunElastix(T1fname_CTRes,InputCT,CTInT1W)
            
            return nibabel.load(CTInT1W)
        else:
            T1WinCT=os.path.splitext(InputT1)[0] + '_InCT.nii.gz'
            RunElastix(InputCT,T1fnameBiasCorrec,T1WinCT)
            return nibabel.load(InputCT)


def BiasCorrecAndCoreg(InputT1,InputZTE):
    #Bias correction
    img = ants.image_read(InputT1)
    img_n4 = ants.n4_bias_field_correction(img)
    img_tmp = img_n4.otsu_segmentation(k=1) # otsu_segmentation
    img_tmp = ants.multi_label_morphology(img_tmp, 'MD', 2) # dilate 2
    img_tmp = ants.smooth_image(img_tmp, 3) # smooth 3
    img_tmp = ants.threshold_image(img_tmp, 0.6) # threshold 0.5
    img_mask = ants.get_mask(img_tmp)
   
    T1fnameBiasCorrec =os.path.splitext(InputT1)[0] + '_BiasCorrec.nii.gz'
    ants.image_write(img_n4,(T1fnameBiasCorrec))
    T1Mask=os.path.splitext(InputT1)[0] + '_SkinMask.nii.gz'
    ants.image_write(img_mask,(T1Mask))
    
    ZTEfnameBiasCorrec=os.path.splitext(InputZTE)[0] + '_BiasCorrec.nii.gz'
    img = ants.image_read(InputZTE)
    img_n4 = ants.n4_bias_field_correction(img)
    ants.image_write(img_n4,(ZTEfnameBiasCorrec))
    
    #coreg

    ZTEInT1W=os.path.splitext(InputZTE)[0] + '_InT1.nii.gz'
    RunElastix(T1fnameBiasCorrec,ZTEfnameBiasCorrec,ZTEInT1W)
    
    img = ants.image_read(T1fnameBiasCorrec)
    img_out = ants.multiply_images(img, img_mask)
    ants.image_write(img_out,(T1fnameBiasCorrec))
    
    img = ants.image_read(ZTEInT1W)
    img_out = ants.multiply_images(img, img_mask)
    ants.image_write(img_out,(ZTEInT1W))
    return T1fnameBiasCorrec,ZTEInT1W, T1Mask

def ConvertZTE_pCT(InputT1,InputZTE,HeadMask,SimbsPath,ThresoldsZTEBone=[0.1,0.6],SimbNIBSType='charm'):
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
