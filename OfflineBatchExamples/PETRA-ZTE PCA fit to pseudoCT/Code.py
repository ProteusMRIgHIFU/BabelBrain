'''
Functions to run PCA fitting between real CT and ZTE/PETRA
Author: Samuel Pichardo. Ph,D.
University of Calgary

'''

import sys
sys.path.append('/Users/spichardo/Documents/GitHub/BabelBrain')
sys.path.append('/Users/spichardo/Documents/GitHub/BabelBrain/BabelBrain')
resource_path = '/Users/spichardo/Documents/GitHub/BabelBrain/BabelBrain'

import numpy as np
import sys
import os
import time
from multiprocessing import Process, Queue
import multiprocessing
import nibabel
import time
import shutil
from glob import glob
from pprint import pprint


from nibabel import processing
from nibabel.spaces import vox2out_vox
import SimpleITK as sitk
import tempfile
import os
import scipy
from scipy import signal
# from CTZTEProcessing import RunElastix
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np
import scipy
from skimage.measure import label, regionprops

from BabelBrain.CalculateMaskProcess import CalculateMaskProcess
from TranscranialModeling.BabelIntegrationBASE import GetSmallestSOS
#we use the Single Element sim setting
from BabelBrain.CalculateFieldProcess import CalculateFieldProcess 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from GPUFunctions.GPUBinaryClosing import BinaryClosing
from os.path import join as jn
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py

class Processing(object):
    '''
    Simple class to organize functions
    '''
    def __init__(self, BasePath,
                      ID,
                      simbnibs_path,
                      Mat4Trajectory,
                      ComputingDevice='6800',
                      ComputingBackend=3, #Metal
                      Frequency=500e3,
                      BasePPW=6,
                      TxSystem='H317',
                      bUseCT=True,
                      CTType=2,
                      ZTE='',
                      ZTERange=[0.1,0.65],
                      HUThreshold=300.0,
                      ElastixOptimizer='FiniteDifferenceGradientDescent'):
        self.BasePath=BasePath
        self.T1W=jn(BasePath,'T1W.nii.gz')
        self.ID=ID
        self.simbnibs_path=simbnibs_path
        self.Mat4Trajectory=Mat4Trajectory
        self.ComputingDevice=ComputingDevice
        self.ComputingBackend=3
        self.Frequency=Frequency
        self.BasePPW=BasePPW
        self.TxSystem=TxSystem
        self.CTType=CTType
        self.ZTE=ZTE
        self.HUThreshold=HUThreshold
        BinaryClosing.InitBinaryClosing(DeviceName=ComputingDevice,GPUBackend='Metal')

    def RunMaskGeneration(self,
                          ZTERange=[0.1,0.65],
                          bReuseFiles=True,
                          ElastixOptimizer='FiniteDifferenceGradientDescent'):
        '''
        This runs the same Step 1 as in BabelBrain GUI
        We need to generate the default pseudo CT that we can use for co-registration purposes
        '''
    
        print("*"*40)
        print("*"*5+" Calculating mask.. BE PATIENT... it can take a couple of minutes...")
        print("*"*40)

        T1W=self.T1W
        ID=self.ID
        simbnibs_path=self.simbnibs_path
        Mat4Trajectory=self.Mat4Trajectory
        deviceName=self.ComputingDevice
        COMPUTING_BACKEND=self.ComputingBackend
        Frequency=self.Frequency
        BasePPW=self.BasePPW
        bUseCT=True
    
        T1WIso= T1W
       
        SmallestSoS= GetSmallestSOS(Frequency,bShear=True)
    
        prefix = ID + '_' + self.TxSystem +'_%ikHz_%iPPW_' %(int(Frequency/1e3),BasePPW)


    
        SpatialStep=np.round(SmallestSoS/Frequency/BasePPW*1e3,3) #step of mask to reconstruct , mm

    
        kargs={}
        kargs['SimbNIBSDir']=simbnibs_path
        kargs['SimbNIBSType']='charm'
        kargs['CoregCT_MRI']=True
        kargs['TrajectoryType']='brainsight'
        kargs['Mat4Trajectory']=Mat4Trajectory
        kargs['T1Source_nii']=T1W
        kargs['T1Conformal_nii']=T1WIso
        # kargs['nIterationsAlign']=10
        kargs['SpatialStep']=SpatialStep
        # kargs['InitialAligment']='HF'
        kargs['Location']=[0,0,0] #This coordinate will be ignored
        kargs['prefix']=prefix
        kargs['bPlot']=False
        kargs['ElastixOptimizer']=ElastixOptimizer
        # kargs['bAlignToSkin']=True
        
        if bUseCT:
            kargs['CT_or_ZTE_input']=self.ZTE
            kargs['CTType']=self.CTType
            if kargs['CTType'] in [2,3]:
                kargs['ZTERange']=ZTERange
            kargs['HUThreshold']=self.HUThreshold
    
        # Start mask generation as a separate process.
        queue=Queue()
        print(COMPUTING_BACKEND,deviceName,kargs)
        maskWorkerProcess = Process(target=CalculateMaskProcess, 
                                    args=(queue,
                                         COMPUTING_BACKEND,
                                         deviceName),
                                    kwargs=kargs)
        maskWorkerProcess.start()      
        # progress.
        T0=time.time()
        bNoError=True
        while maskWorkerProcess.is_alive():
            time.sleep(0.1)
            while queue.empty() == False:
                cMsg=queue.get()
                print(cMsg,end='')
                if '--Babel-Brain-Low-Error' in cMsg:
                    bNoError=False
    
        maskWorkerProcess.join()
        while queue.empty() == False:
            cMsg=queue.get()
            print(cMsg,end='')
            if '--Babel-Brain-Low-Error' in cMsg:
                bNoError=False
        if bNoError:
            TEnd=time.time()
            print('Total time',TEnd-T0)
            print("*"*40)
            print("*"*5+" DONE calculating mask.")
            print("*"*40)
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            raise RuntimeError("Error when generating mask!!")


    def Run_mask_creation(self,ZTERange=[0.1,0.65],bForceRecalculate=False):
        prevfile=os.path.join(self.BasePath,self.ID+'_'+self.TxSystem+'_%ikHz_%iPPW_BabelViscoInput.nii.gz' %(int(self.Frequency/1e3),self.BasePPW))
        if os.path.isfile(prevfile) and bForceRecalculate==False:
            print('skipping generating ',prevfile)
            return
        self.RunMaskGeneration( ZTERange=ZTERange)


    def RunElastix(self,ElastixOptimizer='AdaptiveStochasticGradientDescent',bCToMRI=False):
        '''
        Run BabelBrain's Elastix installation to co-register CT<-->PseudoCT  obtainedi in RunMaskGeneration
        '''
        if bCToMRI:
            reference=jn(self.BasePath,'ZTE_BiasCorrec_pCT.nii.gz')
            moving=jn(self.BasePath,'CT.nii.gz')
            finalname=jn(self.BasePath,'CT_in_T1W.nii.gz')
            tpname=jn(os.path.split(finalname)[0],'ToCT_TransformParameters.txt')
        else:
            reference=jn(self.BasePath,'CT.nii.gz')
            moving=jn(self.BasePath,'ZTE_BiasCorrec_pCT.nii.gz')
            finalname=jn(self.BasePath,'ZTE_pseudo_in_CT.nii.gz')
            tpname=jn(os.path.split(finalname)[0],'ToMRI_TransformParameters.txt')
        
        template =os.path.join(resource_path,'rigid_template.txt')
        with open(template,'r') as g:
            Params=g.readlines()
        #we specify the optimizer to use
        Params.append('\n(Optimizer "'+ElastixOptimizer+'")\n')
        tmpdirname='elastix_output'
        if os.path.isdir(tmpdirname):
            shutil.rmtree(tmpdirname)
        os.mkdir(tmpdirname)
        elastix_param = os.path.join(tmpdirname,'inputparam.txt')
        with open(elastix_param,'w') as g:
            g.writelines(Params)
        shell='zsh'
        path_script = os.path.join(resource_path,"ExternalBin/elastix/run_mac.sh")
        cmd ='"'+path_script + '" "' + reference + '" "' + moving +'" "' + tmpdirname + '" "' + elastix_param + '"'
        print(cmd)
        result = os.system(cmd)
        if result == 0:
            shutil.move(os.path.join(tmpdirname,'result.0.nii.gz'),finalname)
            shutil.move(os.path.join(tmpdirname,'TransformParameters.0.txt'),
                        tpname)
        else:
            raise SystemError("Error when trying to run elastix")
        with open(tpname,'r') as f:
            ln=f.readlines()
        bFound=False
        for n in range(len(ln)):
            if 'ResultImagePixelType' in ln[n]:
                bFound=True
                ln[n]='(ResultImagePixelType "float")\n'
        assert(bFound)
        with open(tpname,'w') as f:
            f.writelines(ln)

    def RunTransformix(self,reference=None,
                      finalname=None):
        if reference is None:
            reference=jn(self.BasePath,'ZTE_BiasCorrec.nii.gz')
            finalname=jn(self.BasePath,'ZTE_in_CT.nii.gz')
        inputtransform=jn(os.path.split(reference)[0],'ToCT_TransformParameters.txt')
        
        tmpdirname='elastix_output'
        if os.path.isdir(tmpdirname):
            shutil.rmtree(tmpdirname)
        os.mkdir(tmpdirname)
        shell='zsh'
        path_script = os.path.join(resource_path,"ExternalBin/elastix/run_mac_transformix.sh")
        cmd ='"'+path_script + '" "' + reference + '" "' + inputtransform +'" "' + tmpdirname +'"'
        print(cmd)
        result = os.system(cmd)
        if result == 0:
            shutil.move(os.path.join(tmpdirname,'result.nii.gz'),finalname)
        else:
            raise SystemError("Error when trying to run elastix")

    def ShowFirstPart(self,LineSep=143):
        '''
        This shows a comparison between CT and pseudoCT, including histogram and density plots.
        Use LineSep to specify the cutout region to ensure only top regions of the skulls are use in the analysis as
        ZTE and CT rarely match in the inferior regions of the head
        '''
        self.LineSep=LineSep
        BasePath=self.BasePath
        ZTE=nibabel.load(jn(BasePath,'ZTE_BiasCorrec_pCT.nii.gz'))
        ZTE_pCT=ZTE.get_fdata()
        CTn=nibabel.load(jn(BasePath,'CT_in_T1W.nii.gz'))
        CT=CTn.get_fdata()
        ZTE_pCT[:,:,:LineSep]=ZTE_pCT.min()
        CT[:,:,:LineSep]=CT.min()
        
        plt.close('all')
        plt.figure(figsize=(10,4))
        ax=plt.subplot(1,2,1)
        vmax=2100
        plt.imshow(ZTE_pCT[ZTE_pCT.shape[0]//2,:,:],vmin=0,vmax=vmax,cmap=plt.cm.gray);plt.colorbar()
        z=ZTE.header.get_zooms()
        ax.set_aspect(z[1]/z[2])
        plt.title('ZTE pseudoCT')
        ax=plt.subplot(1,2,2)
        z=CTn.header.get_zooms()
        plt.imshow(CT[CT.shape[0]//2,:,:],vmin=0,vmax=vmax,cmap=plt.cm.gray);plt.colorbar()
        ax.set_aspect(z[1]/z[2])
        plt.title("Real CT")
        plt.savefig(jn(self.BasePath,'DefaultSettings-CT and pseudo CT.pdf'),bbox_inches='tight')
        
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.hist(ZTE_pCT[ZTE_pCT>300],bins=20,density=True);
        plt.title('ZTE pseudoCT')
        plt.subplot(1,2,2)
        plt.hist(CT[CT>300],bins=20,density=True);
        plt.title("CT")
        plt.savefig(jn(self.BasePath,'DefaultSettings-Histograms.pdf'),bbox_inches='tight')
    
        selpoints =(ZTE_pCT>300) & (CT>300)
        plt.figure()
        plt.hist2d(ZTE_pCT[selpoints], CT[selpoints],bins=40)
        plt.xlabel("pseudoCT")
        plt.ylabel("Real CT")
        plt.savefig(jn(self.BasePath,'DefaultSettings-Density Plot.pdf'),bbox_inches='tight')

    
    def ProcessMRI(self,ZTERange=[0.1,0.65]):
        '''
        Exceute PCA-based analysis to fit ZTE/PETRA and CT
        '''
        BasePath=self.BasePath

        vmax=2100

        bIsPetra = self.CTType ==3

        '''
        We load CT in T1W space (obtained with the RunElastix function)
        and bias corrected ZTE (obtained with RunMaskGeneration)
        '''
        CTnii =nibabel.load(jn(BasePath,'CT_in_T1W.nii.gz'))
        CT=CTnii.get_fdata()
        ZTEnii=nibabel.load(jn(BasePath,'ZTE_InT1.nii.gz'))
        ZTE=ZTEnii.get_fdata()

        '''
        We do the smae processing as in BabelBrain to normalize the ZTE/PETRA data
        '''
        charm= nibabel.load(self.simbnibs_path+os.sep+'final_tissues.nii.gz')
        charmdata=np.ascontiguousarray(charm.get_fdata())[:,:,:,0]
        arrSkin=charmdata>0 #this mimics what the old headreco does for skin
        arrMask=(charmdata==1) | (charmdata==2) | (charmdata==3) | (charmdata==9) #this mimics what the old headreco does for csf
        label_img=label(charmdata==0)
        regions= regionprops(label_img)
        regions=sorted(regions,key=lambda d: d.area) #we eliminate the large background region
        arrCavities=(label_img!=0) &(label_img!=regions[-1].label)

        arrNorm=ZTE.copy()

        if bIsPetra: # FUN23 Miscouridou et al. Adapted from  https://github.com/ucl-bug/petra-to-ct
            print('Using PETRA specification to convert to pCT')

            #histogram normalization
            #histogram normalization
            if (arrNorm.max()-arrNorm.min())>2**16-1:
                raise ValueError('The range of values in the ZTE file exceeds 2^16')
            edgesin=np.arange(int(arrNorm.min()),int(arrNorm.max())+2)-0.5                   
            hist_vals, edges = np.histogram(arrNorm.flatten().astype(int),bins=edgesin)
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
            arrNorm/=np.max(locs)

            plt.figure()
            plt.plot(bins, hist_vals);
            for ind2 in locs:
                plt.plot([ind2,ind2],[np.min(hist_vals),np.max(hist_vals)])
            plt.xlabel('PETRA Value')
            plt.ylabel('Count')
            plt.title('Image Histogram')
            plt.savefig(jn(self.BasePath,'Processing-Normalized PETRA histogram.pdf'),bbox_inches='tight')

        else:
            maskedZTE =arrNorm.copy()
            maskedZTE[arrMask==0]=-1000
            cutoff=np.percentile(maskedZTE[maskedZTE>-500].flatten(),95)
            arrNorm/=cutoff
            arrNorm[arrSkin==0]=-0.5
        
        arrGauss=arrNorm.copy()
        arrGauss[scipy.ndimage.binary_erosion(arrSkin,iterations=3)==0]=np.max(arrGauss)

        '''
        We use the ZTERange input to threshold the region expected for the skull and do morphological operations
        '''
        arr=(arrGauss>=ZTERange[0]) & (arrGauss<=ZTERange[1])
        plt.figure()
        plt.imshow(arr[ZTE.shape[0]//2,:,:]);plt.colorbar()
        plt.savefig(jn(self.BasePath,'Processing-Normalized ZTE-PETRA.pdf'),bbox_inches='tight')

        
        selpoints=arr
        label_img=label(selpoints)
        regions= regionprops(label_img)
        regions=sorted(regions,key=lambda d: d.area)
        selpoints=label_img==regions[-1].label
        sf2=np.round((np.ones(3)*5)).astype(int)
        selpoints=BinaryClosing.BinaryClose(selpoints, structure=np.ones(sf2,dtype=int), GPUBackend='Metal')!=0

        selpoints[:,:,:self.LineSep]=False
        selZTE=selpoints*arrNorm
        selCT=selpoints*CT
        
        selpoints=(selZTE>0)&(selCT>200)
        
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow((selpoints*arrNorm)[arrNorm.shape[0]//2,:,:]);plt.colorbar()
        # plt.subplot(1,2,2)
        # plt.imshow((selpoints*CT)[CT.shape[0]//2,:,:]);plt.colorbar()

        '''
        We normalize data for PCA analysis and run PCA
        '''

        data=np.vstack((arrNorm[selpoints],CT[selpoints])).T
        dataForGGit=data.copy()
        
        sc = StandardScaler()
        data=sc.fit_transform(data)
        X=data
        X_mean = np.mean(X, axis=0)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca.fit(X)


        '''
        We get the PCA first component to obtain a linear fitting function
        '''
        
        # First principal component
        pc1 = pca.components_[0]

        # Compute slope (b/a)
        m = pc1[1] / pc1[0]  # slope of the first principal component
        
        # Compute intercept
        c = X_mean[1] - m * X_mean[0]
        
        # Generate fitted line points
        x_fit = np.linspace(min(X[:,0]), max(X[:,0]), 100)
        y_fit = m * x_fit + c
        
        # # Plot original data and fitted line
        # plt.figure()
        # plt.hist2d(data[:,0],data[:,1],bins=50)
        # plt.plot(x_fit, y_fit, label="PCA Fit", color='red', linestyle='dashed')
        # plt.xlabel("ZTE")
        # plt.ylabel("HU")
        # plt.legend()
        # plt.title("PCA-Based normalized Linear Fit")
        
        
        # Print the final line equation
        xf2=x_fit*sc.scale_[0]+sc.mean_[0]
        yf2=y_fit*sc.scale_[1]+sc.mean_[1]

        if bIsPetra:
            yf3=xf2*(-2929.6)+ 3274.9
        else:
            yf3=xf2*(-2085)+ 2329.0
        
        data=np.vstack((arrNorm[selpoints],CT[selpoints])).T
        plt.figure()
        plt.hist2d(data[:,0],data[:,1],bins=50)
        plt.plot(xf2,yf2,label="PCA Fit", color='blue', linestyle='dashed')
        plt.plot(xf2,yf3,label="Orig.", color='red', linestyle='dashed')
        plt.xlabel("ZTE")
        plt.ylabel("HU")
        plt.legend()
        plt.title("PCA-Based unnormalized Linear Fit")
        plt.savefig(jn(self.BasePath,'Processing-PCA Based unnormalized Linear Fit.pdf'),bbox_inches='tight')

        
        '''
        We convert from PCA normalized condition fitting to un-normalized
        This is the new fitting formula for ZTE/PETRA to pseudo CT
        '''
        lin_fit=np.polyfit(xf2,yf2,1)
        

        label_img = label(arr)
        
        def pixelcount(regionmask):
            return np.sum(regionmask)

        
                
        '''
        We generate the new pseudoCT and do comparisons
        '''
        props = regionprops(label_img, extra_properties=(pixelcount,))
        props = sorted(props, key=itemgetter('pixelcount'), reverse=True)
        regHead=scipy.ndimage.binary_closing(label_img==props[0].label,structure=np.ones((11,11,11))).astype(np.uint8)

        arrCT=np.zeros_like(arrGauss)
        arrCT[arrSkin==0]=-1000 
        arrCT[arrSkin!=0]=42.0 #soft tissue

        # if bIsPetra:
        #     arrCT[regHead!=0]=-2929.6*arrNorm[regHead!=0]+ 3274.9
        # else:
        # arrCT[regHead!=0]=-2085*arrNorm[regHead!=0]+ 2329.0
        arrCT[regHead!=0]=lin_fit[0]*arrNorm[regHead!=0]+ lin_fit[1]

        # arrCT[regHead!=0]=res.slope*arrNorm[regHead!=0]+ res.intercept

        arrCT[arrCT<-1000]=-1000 #air
        arrCT[arrCT>3300]=-1000 #air 
        arrCT[arrCavities!=0]=-1000


        plt.figure(figsize=(10,4))
        ax=plt.subplot(1,2,1)
        vmax=2100
        sarrCT=arrCT.copy()
        sarrCT[:,:,:self.LineSep]=sarrCT.min()
        sCT=CT.copy()
        sCT[:,:,:self.LineSep]=sCT.min()
        plt.imshow(sarrCT[arrCT.shape[0]//2,:,:],vmin=0,vmax=vmax,cmap=plt.cm.gray);plt.colorbar()
        z=ZTEnii.header.get_zooms()
        ax.set_aspect(z[1]/z[2])
        plt.title('New ZTE pseudoCT')
        ax=plt.subplot(1,2,2)
        z=CTnii.header.get_zooms()
        plt.imshow(sCT[sCT.shape[0]//2,:,:],vmin=0,vmax=vmax,cmap=plt.cm.gray);plt.colorbar()
        ax.set_aspect(z[1]/z[2])
        plt.title("Real CT")
        plt.savefig(jn(self.BasePath,'Processing-PCT and new pseudo CT.pdf'),bbox_inches='tight')

        
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.hist(sarrCT[sarrCT>300],bins=20,density=True);
        plt.title('ZTE pseudoCT')
        plt.subplot(1,2,2)
        plt.hist(sCT[sCT>300],bins=20,density=True);
        plt.title("CT")
        plt.savefig(jn(self.BasePath,'Processing-Histograms.pdf'),bbox_inches='tight')


        selpoints =(sarrCT>300) & (sCT>300)
        plt.figure()
        plt.hist2d(sarrCT[selpoints], sCT[selpoints],bins=40)
        plt.xlabel("pseudoCT")
        plt.ylabel("Real CT")
        plt.savefig(jn(self.BasePath,'Processing-Density Plot.pdf'),bbox_inches='tight')
       
        self.NewPCAFit=lin_fit

        '''
        We save the data to combine later for a multi image PCA fitting
        '''

        datatosave={}
        datatosave['NewPCAFit']=lin_fit
        datatosave['CT']=CT
        datatosave['NewPCT']=arrCT
        datatosave['data']=dataForGGit
        datatosave['arrNorm']=arrNorm
        datatosave['selpoints']=selpoints
        datatosave['regHead']=regHead
        datatosave['arrSkin']=arrSkin
        datatosave['arrCavities']=arrCavities
        
        SaveToH5py(datatosave,jn(BasePath,'PCAFitData.h5'))
        
        NewPCT=nibabel.Nifti1Image(arrCT.astype(np.float32),ZTEnii.affine)
        NewPCT.to_filename(jn(BasePath,'New_pseudoCT.nii.gz'))
        

        print(' New PCA fitting',lin_fit)
        


        # data=np.vstack((arrNorm[selpoints],CT[selpoints])).T
        # # Generate sample data (e.g., two correlated variables)
        # np.random.seed(42)
        # x = data[:,0]
        # y = data[:,1]
        
        # # Estimate density using Gaussian Kernel Density Estimation (KDE)
        # xy = np.vstack([x, y])
        # density = gaussian_kde(xy)(xy)
        
        # # Sort data by density (useful for visualization)
        # idx = density.argsort()
        # x, y, density = x[idx], y[idx], density[idx]
        
        # # Stack into a matrix for PCA
        # data = np.vstack([x, y]).T
        
        # # Plot the density scatter and first principal component
        # plt.figure(figsize=(8, 6))
        # plt.scatter(x, y, c=density, cmap='viridis', s=5)
        # plt.colorbar(label="Density")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.legend()
        # plt.title("Density Plot with First Principal Component")
        # plt.show()


        
