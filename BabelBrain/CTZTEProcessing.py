'''
Processing of ZTE traditional (linear) conversion 
Miscouridou, M., Pineda-Pardo, J.A., Stagg, C.J., Treeby, B.E. and Stanziola, A., 
2022. Classical and learned MR to pseudo-CT mappings for accurate transcranial ultrasound simulation.
IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 69(10), pp.2896-2905.
'''
import ants
import nibabel
from nibabel import processing
import itk
import os
import scipy
from skimage.measure import label, regionprops
import numpy as np
from operator import itemgetter

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
        parameters = itk.ParameterObject.New()

        resolutions = 4
        default_rigid = parameters.GetDefaultParameterMap("rigid", resolutions)
        parameters.AddParameterMap(default_rigid)
        if CoregCT_MRI==1:
            #we first upsample the T1W to the same resolution as CT
            fixed_image=nibabel.load(T1fnameBiasCorrec)
            fixed_image=processing.resample_to_output(fixed_image,voxel_sizes=nibabel.load(InputCT).header.get_zooms(),cval=fixed_image.get_fdata().min())
            T1fname_CTRes=os.path.splitext(InputT1)[0] + '_BiasCorrec_CT_res.nii.gz'
            fixed_image.to_filename(T1fname_CTRes)
            fixed_image = itk.imread(T1fname_CTRes,itk.F)
            moving_image = itk.imread(InputCT,itk.F)
            
            CTInT1W=os.path.splitext(InputCT)[0] + '_InT1.nii.gz'
            registered_image, params = itk.elastix_registration_method(fixed_image, moving_image,parameter_object=parameters)
            itk.imwrite(registered_image,CTInT1W)

            return nibabel.load(CTInT1W)
        else:
            fixed_image = itk.imread(InputCT,itk.F)
            moving_image = itk.imread(T1fnameBiasCorrec,itk.F)
            
            T1WinCT=os.path.splitext(InputT1)[0] + '_InCT.nii.gz'
            registered_image, params = itk.elastix_registration_method(fixed_image, moving_image,parameter_object=parameters)
            itk.imwrite(registered_image,T1WinCT)

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
    
    parameters = itk.ParameterObject.New()

    resolutions = 3
    default_rigid = parameters.GetDefaultParameterMap("rigid", resolutions)
    parameters.AddParameterMap(default_rigid)

    fixed_image = itk.imread(T1fnameBiasCorrec,itk.F)
    moving_image = itk.imread(ZTEfnameBiasCorrec,itk.F)
    
    ZTEInT1W=os.path.splitext(InputZTE)[0] + '_InT1.nii.gz'
    registered_image, params = itk.elastix_registration_method(fixed_image, moving_image,parameter_object=parameters)
    itk.imwrite(registered_image,ZTEInT1W)
    
    img = ants.image_read(T1fnameBiasCorrec)
    img_out = ants.multiply_images(img, img_mask)
    ants.image_write(img_out,(T1fnameBiasCorrec))
    
    img = ants.image_read(ZTEInT1W)
    img_out = ants.multiply_images(img, img_mask)
    ants.image_write(img_out,(ZTEInT1W))
    return T1fnameBiasCorrec,ZTEInT1W, T1Mask

def ConvertZTE_pCT(InputT1,InputZTE,HeadMask,SimbsPath,ThresoldsZTEBone=[0.1,0.6]):
    print('converting ZTE to pCT with range',ThresoldsZTEBone)
    InputBrainMask=os.path.join(SimbsPath,'csf.nii.gz')
    SkinMask=os.path.join(SimbsPath,'skin.nii.gz')
    CavitiesMask=os.path.join(SimbsPath,'cavities.nii.gz')
    # Load T1 and ZTE
    volumeT1 = nibabel.load(InputT1)
    volumeZTE = nibabel.load(InputZTE)
    volumeMask = nibabel.load(InputBrainMask)
    volumeHead = nibabel.load(HeadMask)
    volumeSkin = nibabel.load(SkinMask)
    volumeCavities = nibabel.load(CavitiesMask)
    
    
    arrMask=volumeMask.get_fdata()
    arrZTE=volumeZTE.get_fdata()
    arrHead=volumeHead.get_fdata()
    arrSkin=volumeSkin.get_fdata()
    arrCavities=volumeCavities.get_fdata()
    
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
