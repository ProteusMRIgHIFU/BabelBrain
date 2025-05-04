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
import tempfile
import logging
import os
import scipy
from skimage.measure import label, regionprops
import numpy as np
from scipy import signal
from operator import itemgetter
import platform
from pathlib import Path
import sys
import subprocess
import shutil
from linetimer import CodeTimer
import hashlib
import matplotlib.pyplot as plt

logger = logging.getLogger()

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

def GetBlake2sHash(filename):
    # Create a Blake2s hash object of size 4 bytes
    blake2s = hashlib.blake2s(digest_size=4)
    
    # Open the file in binary mode and read it in chunks
    with open(filename, 'rb') as file:
        while True:
            data = file.read(65536)  # Read data in 64KB chunks
            if not data:
                break
            blake2s.update(data)
    
    # Calculate the checksum, a string that is double the hash object size (i.e. 8 bytes)
    checksum = blake2s.hexdigest()
    
    return checksum

def SaveHashInfo(precursorfiles, outputfilename, output=None, CTType=None, HUT=None, ZTER=None):
    # CT type info to be saved
    if CTType == None:
        savedCTType = ""
    elif CTType == 1:
        savedCTType = f"CTType=CT,"
    elif CTType == 2:
        savedCTType = f"CTType=ZTE,"
    else: #3
        savedCTType = f"CTType=PETRA,"
    
    # HU Threshold info to be saved
    if HUT is None:
        savedHUThreshold = ""
    else:
        savedHUThreshold = f"HUT={int(HUT)},"

    # ZTE range info to be saved
    if ZTER is None:
        saveZTERange = ""
    else:
        saveZTERange = f"ZTER={ZTER},"

    # checksum info to be saved
    savedHashes = ""
    for f in precursorfiles:
        savedHashes+=GetBlake2sHash(f)+","
    
    savedInfo = (savedCTType + savedHUThreshold + saveZTERange + savedHashes).encode('utf-8')
    
    if ".nii.gz" in outputfilename:
        if len(savedInfo) <= 80: # Nifiti descrip header can only accept string < 80 bytes (80 utf-8 chars)
            output.header['descrip'] = savedInfo
            nibabel.save(output,outputfilename)
        else:
            print(f"'descrip' string not saved in nifti header as it exceeds text limit ({len(savedInfo)} > 80)")
            nibabel.save(output,outputfilename)
    elif ".npy" in outputfilename: #numpy
        savedInfoNumpy = np.array(savedInfo)
        np.save(outputfilename,savedInfoNumpy)
    else:
        print("Hash data not saved, invalid output file type specified")


def RunElastix(reference,moving,finalname,ElastixOptimizer='AdaptiveStochasticGradientDescent'):
    template =os.path.join(resource_path(),'rigid_template.txt')
    with open(template,'r') as g:
        Params=g.readlines()
    #we specify the optimizer to use
    Params.append('\n(Optimizer "'+ElastixOptimizer+'")\n')
    with tempfile.TemporaryDirectory() as tmpdirname:
        elastix_param = os.path.join(tmpdirname,'inputparam.txt')
        with open(elastix_param,'w') as g:
            g.writelines(Params)
        
        if sys.platform == 'linux' or _IS_MAC:
            if sys.platform == 'linux':
                shell='bash'
                path_script = os.path.join(resource_path(),"ExternalBin/elastix/run_linux.sh")
            elif _IS_MAC:
                shell='zsh'
                path_script = os.path.join(resource_path(),"ExternalBin/elastix/run_mac.sh")
            
            logger.info("Starting Elastix")
            if _IS_MAC:
                cmd ='"'+path_script + '" "' + reference + '" "' + moving +'" "' + tmpdirname + '" "' + elastix_param + '"'
                print(cmd)
                result = os.system(cmd)
            else:
                result = subprocess.run(
                        [shell,
                        path_script,
                        reference,
                        moving,
                        tmpdirname,
                        elastix_param], capture_output=True, text=True
                )
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
                result=result.returncode 
        else:
            path_script = os.path.join(resource_path(),"ExternalBin/elastix/run_win.bat")
            
            logger.info("Starting Elastix")
            result = subprocess.run(
                    [path_script,
                    reference,
                    moving,
                    tmpdirname,
                    elastix_param], capture_output=True, text=True,shell=True,
            )
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            result=result.returncode 
        logger.info("Elastix Finished")
        
        if result == 0:
            shutil.move(os.path.join(tmpdirname,'result.0.nii.gz'),finalname)
        else:
            raise SystemError("Error when trying to run elastix")

def N4BiasCorrec(input,file_manager,hashFiles,output=None,shrinkFactor=4,
                convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},bInvertValues=False):
    with CodeTimer("Load nifti via sitk", unit="s"):
        inputImage = file_manager.load_file(input,nifti_load_method='sitk')

    if bInvertValues:
        imarray=sitk.GetArrayFromImage(inputImage)
        imarray-=imarray.max()
        imarray=-imarray
        modified_sitk_image = sitk.GetImageFromArray(imarray)

        # Set the spacing, origin, and direction to match the original image
        modified_sitk_image.SetSpacing(inputImage.GetSpacing())
        modified_sitk_image.SetOrigin(inputImage.GetOrigin())
        modified_sitk_image.SetDirection(inputImage.GetDirection())
        inputImage=modified_sitk_image
        
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
        corrected_image_full_resolution_nib = file_manager.sitk_to_nibabel(corrected_image_full_resolution)
        file_manager.save_file(file_data=corrected_image_full_resolution_nib,filename=output,precursor_files=hashFiles)

    return corrected_image_full_resolution

def CTCorreg(InputT1,file_manager, ElastixOptimizer, ResampleFunc=None, ResampleBackend='OpenCL'):
    # coreg = 0, do not coregister, just load data
    # coreg = 1 , coregister CT-->MRI space
    # coreg = 2 , coregister MRI-->CT space
    InputCT = file_manager.input_files['ExtraScan']

    #Bias correction
    if file_manager.coreg == 0:
        return file_manager.load_file(InputCT)
    else:
        T1fnameBiasCorrec = file_manager.output_files['T1fnameBiasCorrec']
        N4BiasCorrec(InputT1,
                     file_manager,
                     [file_manager.output_files['ReuseSimbNIBS'],
                      file_manager.output_files['Skull_STL'],
                      file_manager.output_files['CSF_STL'],
                      file_manager.output_files['Skin_STL']],
                     T1fnameBiasCorrec)
        
        #coreg
        if file_manager.coreg == 1:
            #we first upsample the T1W to the same resolution as CT
            in_img = file_manager.load_file(T1fnameBiasCorrec)

            voxel_sizes = file_manager.load_file(InputCT).header.get_zooms()
            cval = in_img.get_fdata().min()
            if ResampleFunc is None:
                fixed_image = processing.resample_to_output(in_img,voxel_sizes=voxel_sizes,cval=cval)
            else:
                # Set up for Resample call
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
                
                fixed_image = ResampleFunc(in_img,out_vox_map,cval=cval, GPUBackend=ResampleBackend)


            T1fname_CTRes = file_manager.output_files['T1fname_CTRes']
            file_manager.save_file(file_data=fixed_image,filename=T1fname_CTRes,precursor_files=T1fnameBiasCorrec)

            CTInT1W = file_manager.output_files['CTInT1W']

            # Since we're not saving the elastix result with our file_manager, we need to ensure the prerequisite files are not currently being saved
            file_manager.wait_for_file(T1fname_CTRes)
            RunElastix(T1fname_CTRes,InputCT,CTInT1W,ElastixOptimizer)

            with CodeTimer("Reloading Elastix Output, adding hashes to header, and saving", unit="s"):
                elastixoutput = file_manager.load_file(CTInT1W)
                file_manager.save_file(file_data=elastixoutput,filename=CTInT1W,precursor_files=T1fname_CTRes)
            
            return elastixoutput
        else:
            T1WinCT = file_manager.output_files['T1WinCT']

            # Since we're not saving the elastix result with our file_manager, we need to ensure the prerequisite files are not currently being saved
            file_manager.wait_for_file(T1fnameBiasCorrec)
            RunElastix(InputCT,T1fnameBiasCorrec,T1WinCT,ElastixOptimizer)

            with CodeTimer("Reloading Elastix Output, adding hashes to header, and saving", unit="s"):
                elastixoutput = file_manager.load_file(T1WinCT)
                file_manager.save_file(file_data=elastixoutput,filename=T1WinCT,precursor_files=T1fnameBiasCorrec)

            return elastixoutput


def BiasCorrecAndCoreg(InputT1,
                       img_mask,
                       file_manager,
                       ElastixOptimizer,
                       bIsPetra=False,
                       bInvertZTE=True):
    
    InputZTE = file_manager.input_files['ExtraScan']
    
    #Bias correction
    T1fnameBiasCorrec= file_manager.output_files['T1fnameBiasCorrec']
    N4BiasCorrec(InputT1,
                 file_manager,
                 [file_manager.output_files['ReuseSimbNIBS'],
                  file_manager.output_files['Skull_STL'],
                  file_manager.output_files['CSF_STL'],
                  file_manager.output_files['Skin_STL']],
                 T1fnameBiasCorrec)

    ZTEfnameBiasCorrec= file_manager.output_files['ZTEfnameBiasCorrec']
    if bIsPetra:
        convergence={"iters": [50,40,30,20,10], "tol": 1e-7}
        N4BiasCorrec(InputZTE,file_manager,T1fnameBiasCorrec,ZTEfnameBiasCorrec,convergence=convergence)
    else:
        N4BiasCorrec(InputZTE,file_manager,T1fnameBiasCorrec,ZTEfnameBiasCorrec,bInvertValues=bInvertZTE)
    
    #coreg
    ZTEInT1W = file_manager.output_files['ZTEInT1W']

    # Since we're not saving the elastix result with our file_manager, we need to ensure the prerequisite files are not currently being saved
    file_manager.wait_for_file(T1fnameBiasCorrec)
    file_manager.wait_for_file(ZTEfnameBiasCorrec)
    RunElastix(T1fnameBiasCorrec,ZTEfnameBiasCorrec,ZTEInT1W,ElastixOptimizer)
    
    img = file_manager.load_file(T1fnameBiasCorrec,nifti_load_method='sitk',sitk_dtype=sitk.sitkFloat32)
    try:
        img_out=img*sitk.Cast(img_mask,sitk.sitkFloat32)
    except:
        try:
            print('Error: mask and image do not have the same spacing under tolerance, first attempt to fix')
            img_mask.SetSpacing(img.GetSpacing()) # some weird rounding can occur, so we try again
            img_out=img*sitk.Cast(img_mask,sitk.sitkFloat32)
        except:
            print('Error: mask and image do not have the same spacing under tolerance, second attempt to fix')
            imgnib=file_manager.sitk_to_nibabel(img)
            img_mask=file_manager.sitk_to_nibabel(img_mask)
            img_mask=nibabel.Nifti1Image(img_mask.get_fdata(),affine=imgnib.affine)
            img_mask=file_manager.nibabel_to_sitk(img_mask)
            img_out=img*sitk.Cast(img_mask,sitk.sitkFloat32)

    img_out_nib = file_manager.sitk_to_nibabel(img_out)
    file_manager.save_file(file_data=img_out_nib,
                           filename=T1fnameBiasCorrec,
                           precursor_files = [file_manager.output_files['ReuseSimbNIBS'],
                                              file_manager.output_files['Skull_STL'],
                                              file_manager.output_files['CSF_STL'],
                                              file_manager.output_files['Skin_STL']])

    with CodeTimer("Reloading niftis, adding hashes to header, and saving", unit="s"):
        ZTEfnameBiasCorrecOutput = file_manager.load_file(ZTEfnameBiasCorrec)
        file_manager.save_file(file_data=ZTEfnameBiasCorrecOutput,filename=ZTEfnameBiasCorrec,precursor_files=T1fnameBiasCorrec)

    img = file_manager.load_file(ZTEInT1W,nifti_load_method='sitk')
    img_out = img*sitk.Cast(img_mask,sitk.sitkFloat32)
    img_out_nib = file_manager.sitk_to_nibabel(img_out)
    file_manager.save_file(file_data=img_out_nib,
                           filename=ZTEInT1W,
                           precursor_files=ZTEfnameBiasCorrec)

    return T1fnameBiasCorrec,ZTEInT1W

def ConvertZTE_PETRA_pCT(InputT1,
                         InputZTE,
                         TMaskItk,
                         file_manager,
                         bIsPetra=False,
                         PetraMRIPeakDistance=50,
                         PetraNPeaks=2,
                         bGeneratePETRAHistogram=False,
                         PETRASlope=-2929.6,
                         PETRAOffset=3274.9,
                         ZTESlope=-2085.0,
                         ZTEOffset=2329.0):
    print('converting ZTE/PETRA to pCT with range',file_manager.pseudo_CT_range)

    SimbsPath = file_manager.simNIBS_dir
    SimbNIBSType = file_manager.simNIBS_type

    if SimbNIBSType=='charm':
        #while charm is much more powerful to segment skull regions, we need to calculate the meshes ourselves
        charminput = os.path.join(SimbsPath,'final_tissues.nii.gz')
        charm = file_manager.load_file(charminput)
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
        volumeMask = file_manager.load_file(InputBrainMask)
        volumeSkin = file_manager.load_file(SkinMask)
        volumeCavities = file_manager.load_file(CavitiesMask)

        arrMask=volumeMask.get_fdata()
        arrSkin=volumeSkin.get_fdata()
        arrCavities=volumeCavities.get_fdata()
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        HeadMask=os.path.join(tmpdirname,'tissueregion.nii.gz')
        TMaskItk_nib = file_manager.sitk_to_nibabel(TMaskItk)
        file_manager.save_file(file_data=TMaskItk_nib,filename=HeadMask)
        
        volumeT1 = file_manager.load_file(InputT1)
        volumeZTE = file_manager.load_file(InputZTE)
        volumeHead = file_manager.load_file(HeadMask)
        
        arrZTE=volumeZTE.get_fdata()
        arrHead=volumeHead.get_fdata()

        if bIsPetra: # FUN23 Miscouridou et al. Adapted from  https://github.com/ucl-bug/petra-to-ct
            print('Using PETRA specification to convert to pCT')

            #histogram normalization
            #histogram normalization
            if (arrZTE.max()-arrZTE.min())>2**16-1:
                raise ValueError('The range of values in the ZTE file exceeds 2^16')
            edgesin=np.arange(int(arrZTE.min()),int(arrZTE.max())+2)-0.5                   
            hist_vals, edges = np.histogram(arrZTE.flatten().astype(int),bins=edgesin)
            bins = (edges[1:] + edges[:-1])/2
            bins = bins[1:]
            hist_vals = hist_vals[1:]

            PeakDistance = int(PetraMRIPeakDistance/np.mean(np.diff(bins)))

            pks,_ = signal.find_peaks(hist_vals,distance=PeakDistance)
            locs = bins[pks]
            pks=hist_vals[pks]

            ind=np.argsort(pks)
            ind=ind[::-1][:PetraNPeaks]
            pks=pks[ind]
            locs=locs[ind]
            arrZTE/=np.max(locs)

            if bGeneratePETRAHistogram:
                plt.figure()
                plt.plot(bins, hist_vals);
                for ind2 in locs:
                    plt.plot([ind2,ind2],[np.min(hist_vals),np.max(hist_vals)])
                plt.xlabel('PETRA Value')
                plt.ylabel('Count')
                plt.title('Image Histogram')
                petrahistofname = InputZTE.split('.nii')[0]+'-PETRA_Histogram.pdf'
                plt.savefig(petrahistofname)
                plt.close('all')
        else:
            maskedZTE =arrZTE.copy()
            maskedZTE[arrMask==0]=-1000
            print('Using ZTE specification to convert to pCT') # as done in M. Miscouridou at al., IEEE Trans. Ultrason. Ferroelectr. Freq. Control, vol. 69, no. 10, pp. 2896-2905, Oct. 2022.  doi: 10.1109/TUFFC.2022.3198522
            cutoff=np.percentile(maskedZTE[maskedZTE>-500].flatten(),95)
            arrZTE/=cutoff
            arrZTE[arrHead==0]=-0.5

        arrGauss=arrZTE.copy()
        arrGauss[scipy.ndimage.binary_erosion(arrHead,iterations=3)==0]=np.max(arrGauss)
        arr=(arrGauss>=file_manager.pseudo_CT_range[0]) & (arrGauss<=file_manager.pseudo_CT_range[1])
            
        label_img = label(arr)
        
        def pixelcount(regionmask):
            return np.sum(regionmask)
        
        props = regionprops(label_img, extra_properties=(pixelcount,))
        props = sorted(props, key=itemgetter('pixelcount'), reverse=True)
        arr2=scipy.ndimage.binary_closing(label_img==props[0].label,structure=np.ones((11,11,11))).astype(np.uint8)

        arrCT=np.zeros_like(arrGauss)
        arrCT[arrSkin==0]=-1000 
        arrCT[arrSkin!=0]=42.0 #soft tissue

        if bIsPetra:
            print('PETRA conversion with slope and offset',PETRASlope,PETRAOffset)
            arrCT[arr2!=0]=PETRASlope*arrZTE[arr2!=0]+ PETRAOffset
        else:
            print('ZTE conversion with slope and offset',ZTESlope,ZTEOffset)
            arrCT[arr2!=0]=ZTESlope*arrZTE[arr2!=0]+ ZTEOffset

        arrCT[arrCT<-1000]=-1000 #air
        arrCT[arrCT>3300]=-1000 #air 
        arrCT[arrCavities!=0]=-1000
        
        pCT = nibabel.Nifti1Image(arrCT,affine=volumeZTE.affine)
        CTfname = file_manager.output_files['pCTfname']
        file_manager.save_file(file_data=pCT,filename=CTfname,precursor_files=file_manager.output_files['ZTEInT1W'])
    return pCT
