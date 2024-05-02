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


def RunElastix(reference,moving,finalname):
    if sys.platform == 'linux' or _IS_MAC:
        if sys.platform == 'linux':
            shell='bash'
            path_script = os.path.join(resource_path(),"ExternalBin/elastix/run_linux.sh")
        elif _IS_MAC:
            shell='zsh'
            path_script = os.path.join(resource_path(),"ExternalBin/elastix/run_mac.sh")
        if _IS_MAC:
            with tempfile.TemporaryDirectory() as tmpdirname:
                cmd ='"'+path_script + '" "' + reference + '" "' + moving +'" "' + tmpdirname + '"'
                print(cmd)
                result = os.system(cmd)
                if result == 0:
                    shutil.move(os.path.join(tmpdirname,'result.0.nii.gz'),finalname)
            if result != 0:
                raise SystemError("Error when trying to run elastix")
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                result = subprocess.run(
                        [shell,
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
    else:
        path_script = os.path.join(resource_path(),"ExternalBin/elastix/run_win.bat")
        with tempfile.TemporaryDirectory() as tmpdirname:
            result = subprocess.run(
                    [path_script,
                    reference,
                    moving,
                    tmpdirname], capture_output=True, text=True,shell=True,
            )
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            if result.returncode == 0:
                shutil.move(os.path.join(tmpdirname,'result.0.nii.gz'),finalname)

        if result.returncode != 0:
            raise SystemError("Error when trying to run elastix")

def N4BiasCorrec(input,hashFiles,output=None,shrinkFactor=4,
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

        # Reload nifti and save info in header
        corrected_image_full_resolution_nifti = nibabel.load(output)
        SaveHashInfo(hashFiles,output,corrected_image_full_resolution_nifti)

    return corrected_image_full_resolution

def CTCorreg(InputT1,InputCT, outputfnames,CoregCT_MRI=0, bReuseFiles=False, ResampleFunc=None, ResampleBackend='OpenCL'):
    # CoregCT_MRI =0, do not coregister, just load data
    # CoregCT_MRI =1 , coregister CT-->MRI space
    # CoregCT_MRI =2 , coregister MRI-->CT space
    #Bias correction
    if CoregCT_MRI==0:
        return nibabel.load(InputCT)
    else:
        T1fnameBiasCorrec= outputfnames['T1fnameBiasCorrec']
        N4BiasCorrec(InputT1,[outputfnames['ReuseSimbNIBS'],outputfnames['Skull_STL'],outputfnames['CSF_STL'],outputfnames['Skin_STL']],T1fnameBiasCorrec)
        
        #coreg
        if CoregCT_MRI==1:
            #we first upsample the T1W to the same resolution as CT
            in_img=nibabel.load(T1fnameBiasCorrec)

            if ResampleFunc is None:
                fixed_image=processing.resample_to_output(in_img,voxel_sizes=nibabel.load(InputCT).header.get_zooms(),cval=in_img.get_fdata().min())
            else:
                # Set up for Resample call
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


            T1fname_CTRes=outputfnames['T1fname_CTRes']
            SaveHashInfo([outputfnames['T1fnameBiasCorrec']],T1fname_CTRes,fixed_image,CTType=1)

            CTInT1W=outputfnames['CTInT1W']

            RunElastix(outputfnames['T1fname_CTRes'],InputCT,CTInT1W)

            with CodeTimer("Reloading Elastix Output, adding hashes to header, and saving", unit="s"):
                elastixoutput = nibabel.load(CTInT1W)
                SaveHashInfo([T1fname_CTRes],CTInT1W,elastixoutput,CTType=1)
            
            return elastixoutput
        else:
            T1WinCT=outputfnames['T1WinCT']
            RunElastix(InputCT,T1fnameBiasCorrec,T1WinCT)

            with CodeTimer("Reloading Elastix Output, adding hashes to header, and saving", unit="s"):
                elastixoutput = nibabel.load(T1WinCT)
                SaveHashInfo([T1fnameBiasCorrec],T1WinCT,elastixoutput,CTType=1)

            return elastixoutput


def BiasCorrecAndCoreg(InputT1,InputZTE,img_mask, outputfnames):
    #Bias correction
    
    T1fnameBiasCorrec= outputfnames['T1fnameBiasCorrec']
    N4BiasCorrec(InputT1,[outputfnames['ReuseSimbNIBS'],outputfnames['Skull_STL'],outputfnames['CSF_STL'],outputfnames['Skin_STL']],T1fnameBiasCorrec)

    ZTEfnameBiasCorrec= outputfnames['ZTEfnameBiasCorrec']
    N4BiasCorrec(InputZTE,[T1fnameBiasCorrec],ZTEfnameBiasCorrec)
    
    #coreg
    ZTEInT1W=outputfnames['ZTEInT1W']

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

    with CodeTimer("Reloading niftis, adding hashes to header, and saving", unit="s"):
        T1fnameBiasCorrecOutput = nibabel.load(T1fnameBiasCorrec)
        ZTEfnameBiasCorrecOutput = nibabel.load(ZTEfnameBiasCorrec)
        ZTEInT1WOutput = nibabel.load(ZTEInT1W)
        SaveHashInfo([outputfnames['ReuseSimbNIBS'],outputfnames['Skull_STL'],outputfnames['CSF_STL'],outputfnames['Skin_STL']],T1fnameBiasCorrec,T1fnameBiasCorrecOutput)
        SaveHashInfo([T1fnameBiasCorrec],ZTEfnameBiasCorrec,ZTEfnameBiasCorrecOutput)
        SaveHashInfo([ZTEfnameBiasCorrec],ZTEInT1W,ZTEInT1WOutput)

    return T1fnameBiasCorrec,ZTEInT1W

def ConvertZTE_PETRA_pCT(InputT1,InputZTE,TMaskItk,SimbsPath,outputfnames,ThresoldsZTEBone=[0.1,0.6],SimbNIBSType='charm',bIsPetra=False,
            PetraMRIPeakDistance=50,PetraNPeaks=2):
    print('converting ZTE/PETRA to pCT with range',ThresoldsZTEBone)

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

        if bIsPetra: # FUN23 Miscouridou et al. Adapted from  https://github.com/ucl-bug/petra-to-ct
            print('Using PETRA specification to convert to pCT')

            #histogram normalization
            hist_vals, edges = np.histogram(arrZTE.flatten().astype(int),bins='auto')
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
        else:
            maskedZTE =arrZTE.copy()
            maskedZTE[arrMask==0]=-1000
            print('Using ZTE specification to convert to pCT') # as done in M. Miscouridou at al., IEEE Trans. Ultrason. Ferroelectr. Freq. Control, vol. 69, no. 10, pp. 2896-2905, Oct. 2022.  doi: 10.1109/TUFFC.2022.3198522
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

        if bIsPetra:
            arrCT[arr2!=0]=-2929.6*arrZTE[arr2!=0]+ 3274.9
        else:
            arrCT[arr2!=0]=-2085*arrZTE[arr2!=0]+ 2329.0

        arrCT[arrCT<-1000]=-1000 #air
        arrCT[arrCT>3300]=-1000 #air 
        arrCT[arrCavities!=0]=-1000
        
        pCT = nibabel.Nifti1Image(arrCT,affine=volumeZTE.affine)
        CTfname=outputfnames['pCTfname']
        SaveHashInfo([outputfnames['ZTEInT1W']],CTfname,pCT,CTType=(bIsPetra+2))
    return pCT
