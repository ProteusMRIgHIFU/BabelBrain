'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

'''
import numpy as np
np.seterr(divide='raise')
import platform
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from BabelViscoFDTD.H5pySimple import ReadFromH5py,SaveToH5py
from BabelViscoFDTD.PropagationModel import PropagationModel
from BabelViscoFDTD.tools.RayleighAndBHTE import InitCuda,InitOpenCL, InitMetal
import nibabel
import SimpleITK as sitk
from scipy import interpolate
import warnings
import time
import gc
import os
import os
import pandas as pd
import h5py
from linetimer import CodeTimer
import pwlf

try:
    import mkl_fft as fft
except:
    print('mkl_fft not available')
    from numpy import fft
    
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

PModel=PropagationModel()
_IS_MAC = platform.system() == 'Darwin'
def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'TranscranialModeling'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

## Global definitions

DbToNeper=1/(20*np.log10(np.exp(1)))

_MapPichardo = ReadFromH5py(os.path.join(resource_path(), 'MapPichardo.h5'))
_PichardoSOS=interpolate.interp2d(_MapPichardo['rho'], _MapPichardo['freq'], _MapPichardo['MapSoS'])
_PichardoAtt=interpolate.interp2d(_MapPichardo['rho'], _MapPichardo['freq'], _MapPichardo['MapAtt'])

def FitSpeedCorticalShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
    FRef=np.array([270e3,836e3])
    Cs270=np.array([1577.0,1498.0,1313.0]).mean()
    Cs836=np.array([1758.0,1674.0,1545.0]).mean()
    CsRef=np.array([Cs270,Cs836])
    p=np.polyfit(FRef, CsRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitSpeedTrabecularShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
    FRef=np.array([270e3,836e3])
    Cs270=np.array([1227.0,1365.0,1200.0]).mean()
    Cs836=np.array([1574.0,1252.0,1327.0]).mean()
    CsRef=np.array([Cs270,Cs836])
    p=np.polyfit(FRef, CsRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def PorosityToSSoS(Phi,frequency):
    sMin=FitSpeedTrabecularShear(frequency)
    sMax=FitSpeedCorticalShear(frequency)
    sSoS = sMin * Phi + sMax*(1.0-Phi)
    return sSoS

def FitAttBoneShear(frequency,reductionFactor=1.0):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
    PichardoData=(57.0/.27 +373/0.836)/2
    return np.round(PichardoData*(frequency/1e6)*reductionFactor) 

def FitSpeedCorticalLong(frequency):
    #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014 
    FRef=np.array([270e3,836e3])
    ClRef=np.array([2448.0,2516])
    p=np.polyfit(FRef, ClRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitSpeedTrabecularLong(frequency):
    #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    FRef=np.array([270e3,836e3])
    ClRef=np.array([2140.0,2300])
    p=np.polyfit(FRef, ClRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitAttCorticalLong_Goss(frequency,reductionFactor=1):
    #from J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    JasaAtt1MHz=(2.15+1.67)/2*100*reductionFactor
    return np.round(JasaAtt1MHz*(frequency/1e6)) 

def FitAttTrabecularLong_Goss(frequency,reductionFactor=1):
    #from J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    JasaAtt1MHz=1.5*100*reductionFactor
    return np.round(JasaAtt1MHz*(frequency/1e6)) 

def FitAttCorticalLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
    # fitting from data obtained from
    #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
    
    return np.round(203.25090263*((frequency/1e6)**bcoeff)*reductionFactor)

def FitAttTrabecularLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
    #reduction factor 
    # fitting from data obtained from
    #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
    return np.round(202.76362433*((frequency/1e6)**bcoeff)*reductionFactor) 

MatFreq={}
for f in np.arange(100e3,1025e3,5e3):
    Material={}
    #Density (kg/m3), LongSoS (m/s), ShearSoS (m/s), Long Att (Np/m), Shear Att (Np/m)
    Material['Water']=     np.array([1000.0, 1500.0, 0.0   ,   0.0,                   0.0] )
    Material['Cortical']=  np.array([1896.5, FitSpeedCorticalLong(f), 
                                             FitSpeedCorticalShear(f),  
                                             FitAttCorticalLong_Multiple(f)  , 
                                             FitAttBoneShear(f)])
    Material['Trabecular']=np.array([1738.0, FitSpeedTrabecularLong(f),
                                             FitSpeedTrabecularShear(f),
                                             FitAttTrabecularLong_Multiple(f) , 
                                             FitAttBoneShear(f)])
    Material['Skin']=           np.array([1116.0, 1537.0, 0.0   ,  2.3*f/500e3 , 0])
    Material['Brain']=          np.array([1041.0, 1562.0, 0.0   ,  3.45*f/500e3 , 0])
   
    #for gm and wm attenuation, average of these 2 reports
    # white matter	0.558	dB/cm/MHz	0.25-0.75				fit to line for ITIS Foundation from .25 to .75 MHz, intercept=0
    # white matter	1.21	dB/cm/MHz 	3.5 to 10		20C		Labuda (2022) - From sectional 2D maps
    # gray matter	0.094	dB/cm/MHz	0.25-0.75				fit to line for ITIS Foundation from .25 to .75 MHz, intercept=0
    # gray matter	0.67	dB/cm/MHz 	3.5 to 10		20C		Labuda (2022) - From sectional 2D maps

    #Labuda et al. 2022 for SoS and attenuation, ITIS for density 
    Material['WhiteMatter']=    np.array([1041.0, 1537.0, 0.0   ,  10.1772968*f/1000e3 , 0])
    Material['GrayMatter']=     np.array([1045.0, 1520.0, 0.0   ,  4.397881647*f/1000e3 , 0])
    Material['CSF']=            np.array([1007.0, 1507.0, 0.0   , 0.0990*f/1000e3 , 0])

    MatFreq[f]=Material


def GetSmallestSOS(frequency,bShear=False):
    SelFreq=MatFreq[frequency]
    SoS=SelFreq['Water'][1]
    for k in SelFreq:
        if  SelFreq[k][1]<SoS:
            SoS=SelFreq[k][1]
        if SelFreq[k][2]>0 and SelFreq[k][2] < SoS:
            SoS=SelFreq[k][2]
    
    if bShear:
        SoS=np.min([SoS,DensityToSSoSPichardo(1000.0)])
        print('GetSmallestSOS',SoS)
    return SoS

def LLSoSITRUST(density):
    return density*1.33 + 167  #

def LATTITRUST_Pinton(frequency):
    att=270*0.1151277918# Np/m/MHz # Med Phys. 2012 Jan;39(1):299-307.doi: 10.1118/1.3668316. 
    return att*frequency/1e6
     
def SATTITRUST_Pinton(frequency):
    att=540*0.1151277918# Np/m/MHz # Med Phys. 2012 Jan;39(1):299-307.doi: 10.1118/1.3668316. 
    return att*frequency/1e6


def primeCheck(n):
    # 0, 1, even numbers greater than 2 are NOT PRIME
    if n==1 or n==0 or (n % 2 == 0 and n > 2):
        return False
    else:
        # Not prime if divisible by another number less
        # or equal to the square root of itself.
        # n**(1/2) returns square root of n
        for i in range(3, int(n**(1/2))+1, 2):
            if n%i == 0:
                return False
        return True
    

def DensityToHUBony(CTIn):
    CTtoDensity=np.array([[-9.47030278e+02,  1.22500000e+00],
       [ 5.20388482e+01,  1.06000000e+03],
       [ 2.02749650e+02,  1.16000000e+03],
       [ 8.10468261e+02,  1.53000000e+03],
       [ 1.00399419e+03,  1.66000000e+03],
       [ 1.23490136e+03,  1.82000000e+03],
       [ 1.41901214e+03,  1.99000000e+03],
       [ 1.65990448e+03,  2.15000000e+03]])
    pwf=pwlf.PiecewiseLinFit(CTtoDensity[:,1],CTtoDensity[:,0])
    pwf.fit_with_breaks(CTtoDensity[:,1])
    return pwf.predict(CTIn)
    

def HUtoDensityKWave(HUin):
    #Adapted from hounsfield2density.m fromk k_wave
    # Phys. Med. Biol., 41, pp. 111-124 (1996).
    # Acoust. Res. Lett. Online, 1(2), pp. 37-42 (2000). 
    HU = HUin+1000
    density = np.zeros_like(HU)

    # apply conversion in several parts using linear fits to the data
    # Part 1: Less than 930 Hounsfield Units
    density[HU < 930] = np.poly1d([1.025793065681423, -5.680404011488714])(HU[HU < 930])

    # Part 2: Between 930 and 1098 (soft tissue region)
    density[(HU >= 930) & (HU <= 1098)] =  np.poly1d([0.9082709691264, 103.6151457847139])(HU[(HU >= 930) & (HU <= 1098)])

    # Part 3: Between 1098 and 1260 (between soft tissue and bone)
    density[(HU > 1098) & (HU < 1260)] =  np.poly1d([0.5108369316599, 539.9977189228704])(HU[(HU > 1098) & (HU < 1260)])

    # Part 4: Greater than 1260 (bone region)
    density[HU >= 1260] =  np.poly1d([0.6625370912451, 348.8555178455294])(HU[HU >= 1260])
    return density

def HUtoDensityAirTissue(HUIn):
    # linear fitting using
    # DensityAir=1.293 
    # DensityTissue=1041
    # HUAir=-1000
    # HUTissue=27
    
    pf=np.array([1.01237293, 1.01366593e+03])
    return np.polyval(pf,HUIn)

def HUtoDensityMarsac(HUin):
    rhomin=1000.0
    rhomax=2700.0
    return rhomin+ (rhomax-rhomin)*HUin/HUin.max()

def HUtoDensityUCLLowEnergy(HuIn):
    #using calibration reported in https://github.com/ucl-bug/petra-to-ct 
    f = h5py.File(os.path.join(resource_path(),'ct-calibration-low-dose-30-March-2023-v1.h5'),'r')
    ct_calibration=f['ct_calibration'][:][0,:,:].T
    return np.interp(HuIn,ct_calibration[0,:],ct_calibration[1,:])

def DensitytoLSOSMarsac(Density):
    cmin=1500.0
    cmax=3000.0
    return cmin+ (cmax-cmin)*(Density-Density.min())/(Density.max()-Density.min())

def DensityToLAttMcDannold(Density,Frequency):
    FreqReference =660e3
    poly=np.flip(np.array([5.71e3,-9.02, 5.40e-3,-1.41e-6,1.36e-10]))
    return np.polyval(poly,Density)*Frequency/FreqReference #we assume a linear relatinship

def DensityToLSOSMcDannold(Density):
    poly=np.flip(np.array([1.24e-3,-7.63e-7,1.69e-10,5.31e-16,-2.79e-18]))
    return 1.0/np.polyval(poly,Density)

def HUtoPorosity(HUin):
    Phi = 1.0 - HUin/HUin.max()
    return Phi

def PorositytoDensity(Phi):
    Density = 1000.0 * Phi + 2200*(1.0-Phi)
    return Density

def PorositytoLSOS(Phi):
    SoS = 1500 * Phi + 3100*(1.0-Phi)
    return SoS

def PorositytoLAtt(Phi,Frequency):
    amin= 2.302555836 *Frequency/1e6
    amax= 92.10223344 *Frequency/1e6
    Att = amin + (amax - amin)*(Phi**0.5)
    return Att

def HUtoAttenuationWebb(HU,Frequency,Params=['GE','120','B','','0.49, 0.63']):
    #these values are for 120 kVp, BonePlus Kernel, axial res = 0.49, slice res=0.63 in GE Scanners
    #Table IV in Webb et al. IEEE Trans Ultrason Ferroelectr Freq Control 68, no. 5 (2020): 1532-1545.
    # DOI: 10.1109/TUFFC.2020.3039743

    lst_str_cols = ['Scanner','Energy','Kernel','Other','Res']
    dict_dtypes = {x : 'str'  for x in lst_str_cols}

    df=pd.read_csv(os.path.join(resource_path(),'WebbHU_Att.csv'),keep_default_na=False,index_col=lst_str_cols,dtype=dict_dtypes)
    
    sel=df.loc[[Params]]

    print('Using Webb Att mapping with Params (Alpha_0, Beta, c)', 
          Params, 
          sel.iloc[0]['Alpha_0']*100,
          sel.iloc[0]['Beta'],
          sel.iloc[0]['c'])

    return (sel.iloc[0]['Alpha_0']*(Frequency/1e6)**sel.iloc[0]['Beta'] * np.exp(HU*(sel.iloc[0]['c'])))*100

def SpeedofSoundWebbDataset():
    lst_str_cols = ['Scanner','Energy','Kernel','Other','Res']
    dict_dtypes = {x : 'str'  for x in lst_str_cols}

    df=pd.read_csv(os.path.join(resource_path(),'WebbHU_SoS.csv'),keep_default_na=False,index_col=lst_str_cols,dtype=dict_dtypes)
    return df


def HUtoLongSpeedofSoundWebb(HU,Params=['GE','120','B','','0.5, 0.6']):
    #these values are for 120 kVp, BonePlus Kernel, axial res = 0.49, slice res=0.63 in GE Scanners
    #Tables I and II in Webb et al. IEEE Trans Ultrason Ferroelectr Freq Control. 2018 Jul; 65(7): 1111–1124. 
    # DOI: 10.1109/TUFFC.2018.2827899

    df=SpeedofSoundWebbDataset()
    
    sel=df.loc[[Params]]

    print('Using Webb  SOS mapping with Params (slope, intercept)', 
          Params, 
          sel.iloc[0]['Slope'],
          sel.iloc[0]['Intercept']*1000.0)

    return sel.iloc[0]['Slope']*HU + sel.iloc[0]['Intercept']*1000.0


def DensityToLSOSPichardo(Density,Frequency):
    return _PichardoSOS(Density,Frequency/1e6)

def DensityToLAttPichardo(Density,Frequency):
    return _PichardoAtt(Density,Frequency/1e6)

def DensityToSSoSPichardo(density):
    #using Physics in Medicine & Biology, vol. 62, bo. 17,p 6938, 2017, we average the values for the two reported frequencies
    return density*0.422 + 680.515  
    

def SaveNiftiEnforcedISO(nii,fn):
    nii.to_filename(fn)
    newfn=fn.split('__.nii.gz')[0]+'.nii.gz'
    res = float(np.round(np.array(nii.header.get_zooms()).mean(),5))
    try:
        pre=sitk.ReadImage(fn)
        pre.SetSpacing([res,res,res])
        sitk.WriteImage(pre, newfn)
        os.remove(fn)
    except:
        res = '%6.5f' % (res)
        cmd='flirt -in "'+fn + '" -ref "'+ fn + '" -applyisoxfm ' +res + ' -nosearch -out "' +fn.split('__.nii.gz')[0]+'.nii.gz'+'"'
        print(cmd)
        assert(os.system(cmd)==0)
        os.remove(fn)

def ResaveNormalized(RPath,Mask):
    assert('_Sub.nii.gz' in RPath)
    NRPath=RPath.replace('_Sub.nii.gz','_Sub_NORM.nii.gz')

    Results=nibabel.load(RPath)

    ResultsData=Results.get_fdata()
    MaskData=Mask.get_fdata()
    ii,jj,kk=np.mgrid[0:ResultsData.shape[0],0:ResultsData.shape[1],0:ResultsData.shape[2]]

    Indexes=np.c_[(ii.flatten().T,jj.flatten().T,kk.flatten().T,np.ones((kk.size,1)))].T

    PosResults=Results.affine.dot(Indexes)

    IndexesMask=np.round(np.linalg.inv(Mask.affine).dot(PosResults)).astype(int)

    SubMask=MaskData[IndexesMask[0,:],IndexesMask[1,:],IndexesMask[2,:]].reshape(ResultsData.shape)
    ResultsData[SubMask<4]=0
    ResultsData/=ResultsData.max()
    NormalizedNifti=nibabel.Nifti1Image(ResultsData,Results.affine,header=Results.header)
    NormalizedNifti.to_filename(NRPath)
    
####
bGPU_INITIALIZED = False
###

class RUN_SIM_BASE(object):
    def CreateSimObject(self,**kargs):
        #this passes extra parameters needed for a given Tx
        raise NotImplementedError("Need to implement this")

    def RunCases(self,targets=[''],deviceName='A6000',COMPUTING_BACKEND=1,
                ID='LIFU1-01',
                basedir='../LIFU Clinical Trial Data/Participants/',
                bTightNarrowBeamDomain=True,
                TxMechanicalAdjustmentX=0,
                TxMechanicalAdjustmentY=0,
                TxMechanicalAdjustmentZ=0,
                basePPW=[9],
                bDoRefocusing=False,
                extrasuffix='',
                Frequencies= [700e3],
                bDisplay=False,
                bMinimalSaving=False,
                bForceRecalc=False,
                bUseCT=False,
                bWaterOnly=False,
                bDryRun=False,
                bUseRayleighForWater=False,
                bSaveStress=False,
                bSaveDisplacement=False,
                bForceHomogenousMedium=False,
                BenchmarkTestFile='',
                OptimizedWeightsFile='',
                HomogenousMediumValues={'Density':1000.0, #kg/m3 
                                    'LongSoS':1500.0, #m/s
                                    'LongAtt':5.0,
                                    'ShearSoS':0.0, #m/s
                                    'ShearAtt':0.0}, #Np/m
                InputFocusStart='',
                **kargs):
        
        global bGPU_INITIALIZED
        
        if not bGPU_INITIALIZED:
            if COMPUTING_BACKEND==1:
                InitCuda(deviceName)
            elif COMPUTING_BACKEND==2:
                InitOpenCL(deviceName)
            elif COMPUTING_BACKEND==3:
                InitMetal(deviceName)
            bGPU_INITIALIZED=True
            
        OutNames=[]
        for target in targets:
            subsamplingFactor=1 
            #sub sample when save the final results.
            for Frequency in Frequencies:
                fstr='_%ikHz_' %(int(Frequency/1e3))
                
                AlphaCFL=0.5
                for PPW in basePPW:
                    ppws='%iPPW_' % PPW
                    SensorSubSampling=0 #this will force to recalculaet

                    prefix=basedir+ID+os.sep
                    MASKFNAME=prefix+target+fstr+ppws+ 'BabelViscoInput.nii.gz'
                    
                    print (MASKFNAME)
                    if bUseCT:
                        CTFNAME=prefix+target+fstr+ppws+ 'CT.nii.gz'
                    else:
                        CTFNAME=None

                    FILENAMES=OutputFileNames(MASKFNAME,target,Frequency,PPW,extrasuffix,bWaterOnly)
                    FILENAMESWater=None
                    if bUseRayleighForWater:
                        # we store also the filenames for water only
                        FILENAMESWater=OutputFileNames(MASKFNAME,target,Frequency,PPW,extrasuffix,True)
                    cname=FILENAMES['DataForSim']
                    print(cname)
                    OutNames.append(cname)
                    if (os.path.isfile(cname)and not bForceRecalc):
                        print('*'*50)
                        print (' Skipping '+ cname)
                        print('*'*50)
                        continue
                    
                    if bDryRun:
                        #we just need to calculate the filenames
                        continue

                    TestClass=self.CreateSimObject(MASKFNAME=MASKFNAME,
                                                    bTightNarrowBeamDomain=bTightNarrowBeamDomain,
                                                    Frequency=Frequency,
                                                    basePPW=PPW,
                                                    SensorSubSampling=SensorSubSampling,
                                                    AlphaCFL=AlphaCFL,
                                                    bWaterOnly=bWaterOnly,
                                                    TxMechanicalAdjustmentX=TxMechanicalAdjustmentX,
                                                    TxMechanicalAdjustmentY=TxMechanicalAdjustmentY,
                                                    TxMechanicalAdjustmentZ=TxMechanicalAdjustmentZ,
                                                    bDoRefocusing=bDoRefocusing,
                                                    CTFNAME=CTFNAME,
                                                    bDisplay=bDisplay,
                                                    bSaveStress=bSaveStress,
                                                    bSaveDisplacement=bSaveDisplacement,
                                                    bForceHomogenousMedium=bForceHomogenousMedium,
                                                    HomogenousMediumValues=HomogenousMediumValues,
                                                    BenchmarkTestFile=BenchmarkTestFile,
                                                    InputFocusStart=InputFocusStart,
                                                    OptimizedWeightsFile=OptimizedWeightsFile,
                                                    **kargs)
                    print('  Step 1')

                    #with suppress_stdout():
                    with CodeTimer("Time for step 1",unit='s'):
                        TestClass.Step1_InitializeConditions()
                    print('  Step 2')
                    with CodeTimer("Time for step 2",unit='s'):
                        TestClass.Step2_CalculateRayleighFieldsForward(prefix=FILENAMES['outName'],
                                                                    deviceName=deviceName,
                                                                    bSkipSavingSTL= bMinimalSaving)

                    print('  Step 3')
                    with CodeTimer("Time for step 3",unit='s'):
                        TestClass.Step3_CreateSourceSignal_and_Sensor()
                    print('  Step 4')
                    with CodeTimer("Time for step 4",unit='s'):
                        TestClass.Step4_Run_Simulation(GPUName=deviceName,COMPUTING_BACKEND=COMPUTING_BACKEND)
                    print('  Step 5')
                    with CodeTimer("Time for step 5",unit='s'):
                        TestClass.Step5_ExtractPhaseDataForwardandBack()
                    if bDoRefocusing:

                        print('  Step 6')
                        with CodeTimer("Time for step 6",unit='s'):
                            TestClass.Step6_BackPropagationRayleigh(deviceName=deviceName)
                        print('  Step 7')
                        with CodeTimer("Time for step 7",unit='s'):
                            TestClass.Step7_Run_Simulation_Refocus(GPUName=deviceName,COMPUTING_BACKEND=COMPUTING_BACKEND)
                        print('  Step 8')
                        with CodeTimer("Time for step 8",unit='s'):
                            TestClass.Step8_ExtractPhaseDataRefocus()
                    print('  Step 9')
                    with CodeTimer("Time for step 9",unit='s'):
                        TestClass.Step9_PrepAndPlotData()
                    print('  Step 10')
                    with CodeTimer("Time for step 10",unit='s'):
                        TestClass.Step10_GetResults(FILENAMES,subsamplingFactor=subsamplingFactor,
                                                        bMinimalSaving=bMinimalSaving,
                                                        bUseRayleighForWater=bUseRayleighForWater,
                                                        FILENAMESWater=FILENAMESWater)
        if 'TEST_FORCE_ERROR_BABEL_STEP2' in os.environ:
            if  os.environ['TEST_FORCE_ERROR_BABEL_STEP2']=='1':
                raise ValueError('TEST_FORCE_ERROR_BABEL_STEP2 was set to 1')
        return OutNames
    
def OutputFileNames(MASKFNAME,target,Frequency,PPW,extrasuffix,bWaterOnly):
    #this create a centralized filenaming of output files that can be used in GUI and in the simulations
    if bWaterOnly:
        waterPrefix='Water_'
    else:
        waterPrefix=''

    bdir=os.path.dirname(MASKFNAME)
    fstr='_%ikHz_' %(int(Frequency/1e3))
    ppws='%iPPW_' % PPW
    
    outName=target+fstr+ppws+extrasuffix
    CPREFIX = bdir+os.sep+outName+waterPrefix
    OUT_FNAMES={}
    OUT_FNAMES['outName']=outName
    OUT_FNAMES['RayleighFreeWaterWOverlay__'] = CPREFIX+'RayleighFreeWaterWOverlay__.nii.gz'
    OUT_FNAMES['RayleighFreeWater__'] = CPREFIX+'RayleighFreeWater__.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocus']=CPREFIX+'FullElasticSolutionRefocus.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocus_Sub']=CPREFIX+'FullElasticSolutionRefocus_Sub.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocus__']=CPREFIX+'FullElasticSolutionRefocus__.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocusPhase__']=CPREFIX+'FullElasticSolutionRefocusPhase__.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocus_Sub__']=CPREFIX+'FullElasticSolutionRefocus_Sub__.nii.gz'
    OUT_FNAMES['FullElasticSolution']=CPREFIX+'FullElasticSolution.nii.gz'
    OUT_FNAMES['FullElasticSolution_Sub']=CPREFIX+'FullElasticSolution_Sub.nii.gz'
    OUT_FNAMES['FullElasticSolution__']=CPREFIX+'FullElasticSolution__.nii.gz'
    OUT_FNAMES['FullElasticSolutionPhase__']=CPREFIX+'FullElasticSolutionPhase__.nii.gz'
    OUT_FNAMES['FullElasticSolution_Sub__']=CPREFIX+'FullElasticSolution_Sub__.nii.gz'
    OUT_FNAMES['DataForSim']=CPREFIX+'DataForSim.h5'
    return OUT_FNAMES

class BabelFTD_Simulations_BASE(object):
    #Meta class dealing with the specificis of each test based on the string name
    def __init__(self,MASKFNAME='',
                 Frequency=250e3,
                 bDisplay=True,
                 basePPW=9,
                 AlphaCFL=1.0,
                 bNoShear=False,
                 pressure=50e3,
                 SensorSubSampling=8,
                 bTightNarrowBeamDomain=False, #if this set, simulations will be done only across a section area that follows the acoustic beam, this is useful to reduce computational costs
                 zLengthBeyonFocalPointWhenNarrow=4e-2,
                 TxMechanicalAdjustmentX=0.0, #Positioning of Tx
                 TxMechanicalAdjustmentY=0.0,
                 TxMechanicalAdjustmentZ=0.0,
                 ExtraAdjustX=[0.0], #these parameters help to enlarge the FOV for any reason (Steering, multipoint, etc.)
                 ExtraAdjustY=[0.0],
                 ZIntoSkin=0.0, # For simulations mimicking compressing skin (in simulation we will remove tissue layers)
                 bDoRefocusing=True,
                 bWaterOnly=False,
                 QCorrection=3,
                 MappingMethod='Webb-Marsac',
                 CTMapCombo=('GE','120','B','','0.5, 0.6'),
                 bPETRA = False, #Specify if CT is derived from PETRA
                 bDensity = False, #Specify if a density map is directly passed
                 CTFNAME=None,
                 bSaveStress=False,
                 bSaveDisplacement=False,
                 bForceHomogenousMedium=False,
                 HomogenousMediumValues={'Density':1000.0, #kg/m3 
                                    'LongSoS':1500.0, #m/s
                                    'LongAtt':25.0, #Np/m
                                    'ShearSoS':0.0, #m/s
                                    'ShearAtt':5.0}, #Np/m
                 BenchmarkTestFile='',
                 InputFocusStart='',
                 OptimizedWeightsFile=''):
        self._MASKFNAME=MASKFNAME
        
        if bNoShear:
            self._Shear=0.0
        else:
            self._Shear=1.0

        self._basePPW=basePPW
        
        self._AlphaCFL=AlphaCFL
        self._bDisplay=bDisplay
        
        self._Frequency=Frequency
        self._pressure=pressure
        self._bWaterOnly=bWaterOnly
        self._bTightNarrowBeamDomain=bTightNarrowBeamDomain
        self._zLengthBeyonFocalPointWhenNarrow=zLengthBeyonFocalPointWhenNarrow
        self._TxMechanicalAdjustmentX=TxMechanicalAdjustmentX
        self._TxMechanicalAdjustmentY=TxMechanicalAdjustmentY
        self._TxMechanicalAdjustmentZ=TxMechanicalAdjustmentZ
        self._ZIntoSkin=ZIntoSkin
        self._bDoRefocusing=bDoRefocusing
        self._SensorSubSampling=SensorSubSampling
        self._CTFNAME=CTFNAME
        self._QCorrection=QCorrection
        self._MappingMethod=MappingMethod
        self._CTMapCombo=CTMapCombo
        self._bPETRA = bPETRA
        self._bDensity = bDensity
        self._ExtraDepthAdjust = 0.0 
        self._ExtraAdjustX = ExtraAdjustX 
        self._ExtraAdjustY = ExtraAdjustY
        self._bSaveStress = bSaveStress
        self._bSaveDisplacement = bSaveDisplacement
        self._bForceHomogenousMedium=bForceHomogenousMedium
        self._HomogenousMediumValues =HomogenousMediumValues
        self._BenchmarkTestFile=BenchmarkTestFile
        self._InputFocusStart=InputFocusStart
        self._OptimizedWeightsFile=OptimizedWeightsFile

    def CreateSimConditions(self,**kargs):
        raise NotImplementedError("Need to implement this")

    def AdjustMechanicalSettings(self,SkullMaskDataOrig,voxelS):
        #in some Tx settings, we adjust here settings of distance
        pass

    def Step1_InitializeConditions(self): #in case it is desired to move up or down in the Z direction the focal spot
        self._SkullMask=nibabel.load(self._MASKFNAME)
        SkullMaskDataOrig=np.flip(self._SkullMask.get_fdata(),axis=2)

        bBrainSegmentation = np.any(SkullMaskDataOrig>5)
        if bBrainSegmentation:
            print('Using segmented white matter, gray matter and CSF')
        voxelS=np.array(self._SkullMask.header.get_zooms())*1e-3
        Dims=np.array(SkullMaskDataOrig.shape)*voxelS
        
        self.AdjustMechanicalSettings(SkullMaskDataOrig,voxelS)

        DensityCTMap=None
        if self._CTFNAME is not None and not self._bWaterOnly\
                                     and not self._bForceHomogenousMedium\
                                     and len(self._BenchmarkTestFile)==0:
            DensityCTMap = np.flip(nibabel.load(self._CTFNAME).get_fdata(),axis=2).astype(np.uint32)
            AllBoneHU = np.load(self._CTFNAME.split('CT.nii.gz')[0]+'CT-cal.npz')['UniqueHU']
            print('Range HU CT, Unique entries',AllBoneHU.min(),AllBoneHU.max(),len(AllBoneHU))
            print('USING MAPPING METHOD = ',self._MappingMethod)
            
            if self._bDensity:
                print('Density map specified, converting Density to HU')
                DensityCTIT= AllBoneHU.copy()
                print('min, max Density',DensityCTIT.min(),DensityCTIT.max())
                AllBoneHU = DensityToHUBony(DensityCTIT)
                print('min, max HU',AllBoneHU.min(),AllBoneHU.max())
            
            Porosity=HUtoPorosity(AllBoneHU)
            if self._MappingMethod=='Webb-Marsac':
                if self._bDensity == False:
                    if self._bPETRA:
                        print('Using PETRA to low energy 70 Kvp CT settings')
                        DensityCTIT=HUtoDensityUCLLowEnergy(AllBoneHU)
                    else:
                        print('Using 120 Kvp CT settings')
                        DensityCTIT=HUtoDensityMarsac(AllBoneHU)
                print('Using CT combination', self._CTMapCombo)
                LSoSIT = HUtoLongSpeedofSoundWebb(AllBoneHU,Params=self._CTMapCombo)
                LAttIT = HUtoAttenuationWebb(AllBoneHU,self._Frequency,Params=self._CTMapCombo)
            elif self._MappingMethod=='Aubry':
                if self._bDensity == False:
                    DensityCTIT = PorositytoDensity(Porosity)
                LSoSIT = PorositytoLSOS(Porosity)
                LAttIT = PorositytoLAtt(Porosity,self._Frequency)
            elif  self._MappingMethod=='Pichardo':
                if self._bDensity == False:
                    DensityCTIT=HUtoDensityAirTissue(AllBoneHU)
                LSoSIT=DensityToLSOSPichardo(DensityCTIT,self._Frequency)
                LAttIT=DensityToLAttPichardo(DensityCTIT,self._Frequency)
            elif self._MappingMethod=='McDannold':
                if self._bDensity == False:
                    DensityCTIT=HUtoDensityAirTissue(AllBoneHU)
                LSoSIT=DensityToLSOSMcDannold(DensityCTIT)
                LAttIT=DensityToLAttMcDannold(DensityCTIT,self._Frequency)
            #these are more experimental
            elif self._MappingMethod=='Marsac-Aubry':
                #Marsac did not calculate attenuation... we use Aubry's old
                if self._bDensity == False:
                    DensityCTIT=HUtoDensityMarsac(AllBoneHU)
                LSoSIT=DensitytoLSOSMarsac(DensityCTIT)
                LAttIT = PorositytoLAtt(AllBoneHU,self._Frequency)
            elif self._MappingMethod=='Pichardo-Marsac':
                #Marsac did not calculate attenuation... we use Aubry's old
                if self._bDensity == False:
                    DensityCTIT=HUtoDensityMarsac(AllBoneHU)
                LSoSIT=DensityToLSOSPichardo(DensityCTIT,self._Frequency)
                LAttIT=DensityToLAttPichardo(DensityCTIT,self._Frequency)
            elif self._MappingMethod=='McDannold-Marsac':
                #Marsac did not calculate attenuation... we use Aubry's old
                if self._bDensity == False:
                    DensityCTIT=HUtoDensityMarsac(AllBoneHU)
                LSoSIT=DensityToLSOSMcDannold(DensityCTIT)
                LAttIT=DensityToLAttMcDannold(DensityCTIT,self._Frequency)
            else:
                raise ValueError('Unknown mapping method -' +self._MappingMethod )
            
            if bBrainSegmentation:
                DensityCTMap+=6  #The material index needs to add 3 to account water, skin, brain (non specific), white matter, gray matter and CSF
            else:
                DensityCTMap+=3 # The material index needs to add 3 to account water, skin and brain
            print("maximum CT index map value",DensityCTMap.max())
            print(" CT Map unique values",np.unique(DensityCTMap).shape)

        #we only adjust Qcorrection for skull material, not for soft tissue
        if self._bWaterOnly:
            QCorrArr =1.0
        elif self._bForceHomogenousMedium:
            QCorrArr = np.ones(2)
        elif len(self._BenchmarkTestFile)>0:
            InputDataBenchmark=ReadFromH5py(self._BenchmarkTestFile)
            assert(len(InputDataBenchmark['Materials'])==len(np.unique(InputDataBenchmark['MaterialMap'])))
            QCorrArr = np.ones(len(InputDataBenchmark['Materials']))
        elif  self._CTFNAME is None:
            if bBrainSegmentation:
                QCorrArr = np.ones(8)
            else:
                QCorrArr = np.ones(5)
        else:
            if bBrainSegmentation:
                QCorrArr=np.ones(6+len(DensityCTIT))
                QCorrArr[6:]=self._QCorrection
            else:
                QCorrArr=np.ones(3+len(DensityCTIT))
                QCorrArr[3:]=self._QCorrection


        self._SIM_SETTINGS = self.CreateSimConditions(baseMaterial=Material['Water'],
                                basePPW=self._basePPW,
                                Frequency=self._Frequency,
                                PaddingForKArray=0,
                                bDisplay=self._bDisplay, 
                                DimDomain=Dims,
                                SensorSubSampling=self._SensorSubSampling,
                                SourceAmp=self._pressure,
                                bTightNarrowBeamDomain=self._bTightNarrowBeamDomain,
                                zLengthBeyonFocalPointWhenNarrow=self._zLengthBeyonFocalPointWhenNarrow,
                                TxMechanicalAdjustmentX=self._TxMechanicalAdjustmentX,
                                TxMechanicalAdjustmentY=self._TxMechanicalAdjustmentY,
                                TxMechanicalAdjustmentZ=self._TxMechanicalAdjustmentZ,
                                ZIntoSkin=self._ZIntoSkin,
                                DensityCTMap=DensityCTMap,
                                QCorrection=QCorrArr,
                                DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721],
                                ExtraDepthAdjust=self._ExtraDepthAdjust,
                                ExtraAdjustX=self._ExtraAdjustX,
                                ExtraAdjustY=self._ExtraAdjustY,
                                bSaveStress=self._bSaveStress,
                                bSaveDisplacement=self._bSaveDisplacement,
                                BenchmarkTestFile=self._BenchmarkTestFile,
                                InputFocusStart=self._InputFocusStart,
                                OptimizedWeightsFile=self._OptimizedWeightsFile)
        
        #####
        ##### bForceHomogenousMedium and BenchmarkTestFile are only for testing
        #####
        
        if self._bForceHomogenousMedium and not self._bWaterOnly:
            print('Forcing using homogenous material with', self._HomogenousMediumValues)
            self._SIM_SETTINGS.AddMaterial(self._HomogenousMediumValues['Density'], #den
                                           self._HomogenousMediumValues['LongSoS'],
                                           self._HomogenousMediumValues['ShearSoS'],
                                           self._HomogenousMediumValues['LongAtt'],
                                           self._HomogenousMediumValues['ShearAtt']) 
        elif len(self._BenchmarkTestFile) > 0 and not self._bWaterOnly:
            print('Forcing using benchmark materials', InputDataBenchmark['Materials'])
            self._SIM_SETTINGS.ResetMaterial() #we remove the default water material
            for entry in InputDataBenchmark['Materials']:
                self._SIM_SETTINGS.AddMaterial(entry['Density'], 
                                           entry['LongSoS'],
                                           entry['ShearSoS'],
                                           entry['LongAtt'],
                                           entry['ShearAtt'])
        elif self._CTFNAME is not None and not self._bWaterOnly:
            lMaterials =['Skin','Brain']
            if bBrainSegmentation:
                lMaterials+=['WhiteMatter','GrayMatter','CSF']
            for k in lMaterials:
                SelM=MatFreq[self._Frequency][k]
                self._SIM_SETTINGS.AddMaterial(SelM[0], #den
                                            SelM[1],
                                            0,
                                            SelM[3],
                                            0) 
            #we disable shear when doing mapping as we need to develop in tandem, otherwise it can end with unrealistic
            # Poison coefficient
            for d,lSoS,lAtt in zip(DensityCTIT,LSoSIT,LAttIT):

                self._SIM_SETTINGS.AddMaterial(d, #den
                                        lSoS,
                                        0,
                                        lAtt,
                                        0)

            
            print('Total MAterials',self._SIM_SETTINGS.ReturnArrayMaterial().shape[0])
                

        elif not self._bWaterOnly:
             lMaterials =['Skin','Cortical','Trabecular','Brain']
             if bBrainSegmentation:
                lMaterials+=['WhiteMatter','GrayMatter','CSF']
             for k in lMaterials:
                SelM=MatFreq[self._Frequency][k]
                Water=MatFreq[self._Frequency]['Water']
                self._SIM_SETTINGS.AddMaterial(SelM[0], #den
                                            SelM[1],
                                            SelM[2]*self._Shear,
                                            SelM[3],
                                            SelM[4]*self._Shear)
        self._SIM_SETTINGS.UpdateConditions(self._SkullMask,
                                            AlphaCFL=self._AlphaCFL,
                                            bWaterOnly=self._bWaterOnly,
                                            bForceHomogenousMedium=self._bForceHomogenousMedium,
                                            BenchmarkTestFile=self._BenchmarkTestFile)
        gc.collect()

    def GenerateSTLTx(self,prefix):
        pass
        
    def Step2_CalculateRayleighFieldsForward(self,prefix='',deviceName='6800',bSkipSavingSTL=False):
        #we use Rayliegh to forward propagate until a plane on top the skull, this plane will be used as a source in BabelVisco
        self._SIM_SETTINGS.CalculateRayleighFieldsForward(deviceName=deviceName)
        if bSkipSavingSTL ==False:
            self.GenerateSTLTx(prefix)
        gc.collect()
        

    def Step3_CreateSourceSignal_and_Sensor(self):
        self._SIM_SETTINGS.CreateSources()
        gc.collect()
        self._SIM_SETTINGS.CreateSensorMap()
        gc.collect()

    def Step4_Run_Simulation(self,GPUName='GP100',bApplyCorrectionForDispersion=True,COMPUTING_BACKEND=1):
        SelMapsRMSPeakList=['Pressure']
        if self._bSaveStress:
            SelMapsRMSPeakList+=['Sigmaxx','Sigmayy', 'Sigmazz']
        if self._bSaveDisplacement:
            SelMapsRMSPeakList+=['Vx','Vy', 'Vz']
        self._SIM_SETTINGS.RUN_SIMULATION(GPUName=GPUName,SelMapsRMSPeakList=SelMapsRMSPeakList,
                                          bApplyCorrectionForDispersion=bApplyCorrectionForDispersion,
                                          COMPUTING_BACKEND=COMPUTING_BACKEND,
                                          bDoRefocusing=self._bDoRefocusing)
        gc.collect()

    def Step5_ExtractPhaseDataForwardandBack(self):
        self._SIM_SETTINGS.CalculatePhaseData(bDoRefocusing=self._bDoRefocusing)
        gc.collect()
        #self._SIM_SETTINGS.PlotResultsPlanePartial()
        
        
    def Step6_BackPropagationRayleigh(self,deviceName='6800'):
        self._SIM_SETTINGS.BackPropagationRayleigh(deviceName=deviceName)
        gc.collect()
        self._SIM_SETTINGS.CreateSourcesRefocus()
        gc.collect()
        
    def Step7_Run_Simulation_Refocus(self,GPUName='GP100',COMPUTING_BACKEND=1,bApplyCorrectionForDispersion=True):
        SelMapsRMSPeakList=['Pressure']
        if self._bSaveStress:
            SelMapsRMSPeakList+=['Sigmaxx','Sigmayy', 'Sigmazz']
        if self._bSaveDisplacement:
            SelMapsRMSPeakList+=['Vx','Vy', 'Vz']
        self._SIM_SETTINGS.RUN_SIMULATION(GPUName=GPUName,
                                          SelMapsRMSPeakList=SelMapsRMSPeakList,
                                          bApplyCorrectionForDispersion=bApplyCorrectionForDispersion,
                                          bRefocused=True,COMPUTING_BACKEND=COMPUTING_BACKEND)
        gc.collect()
    def Step8_ExtractPhaseDataRefocus(self):
        self._SIM_SETTINGS.CalculatePhaseData(bRefocused=True)
        gc.collect()
        
    def Step9_PrepAndPlotData(self):
        self._SIM_SETTINGS.PlotResultsPlane(bDoRefocusing=self._bDoRefocusing)
        gc.collect()
        
    def AddSaveDataSim(self,DataForSim):
        pass

    def Step10_GetResults(self,FILENAMES,subsamplingFactor=1,bMinimalSaving=False,bUseRayleighForWater=False,FILENAMESWater=None):
        ss=subsamplingFactor

        RayleighWater,RayleighWaterOverlay,\
            FullSolutionPressure,\
            FullSolutionPressureRefocus,\
            DataForSim,\
            MaskCalcRegions,\
            FullSolutionPhase,\
            FullSolutionPhaseRefocus,\
            RayleighWaterPhase= self._SIM_SETTINGS.ReturnResults(bDoRefocusing=self._bDoRefocusing,bUseRayleighForWater=bUseRayleighForWater)
        affine=self._SkullMask.affine.copy()
        affineSub=affine.copy()
        affine[0:3,0:3]=affine[0:3,0:3] @ (np.eye(3)*subsamplingFactor)

        if bMinimalSaving==False and not bUseRayleighForWater:
            nii=nibabel.Nifti1Image(RayleighWaterOverlay[::ss,::ss,::ss],affine=affine)
            SaveNiftiEnforcedISO(nii,FILENAMES['RayleighFreeWaterWOverlay__'])
        
        if not bUseRayleighForWater: 
            nii=nibabel.Nifti1Image(RayleighWater[::ss,::ss,::ss],affine=affine)
            SaveNiftiEnforcedISO(nii,FILENAMES['RayleighFreeWater__'])

        [mx,my,mz]=np.where(MaskCalcRegions)
        locm=np.array([[mx[0],my[0],mz[0],1]]).T
        NewOrig=affineSub @ locm
        affineSub[0:3,3]=NewOrig[0:3,0]
        mx=np.unique(mx.flatten())
        my=np.unique(my.flatten())
        mz=np.unique(mz.flatten())
        if self._bDoRefocusing:
            nii=nibabel.Nifti1Image(FullSolutionPressureRefocus[::ss,::ss,::ss],affine=affine)
            SaveNiftiEnforcedISO(nii,FILENAMES['FullElasticSolutionRefocus__'])
            nii=nibabel.Nifti1Image(FullSolutionPhaseRefocus[::ss,::ss,::ss],affine=affine)
            SaveNiftiEnforcedISO(nii,FILENAMES['FullElasticSolutionRefocusPhase__'])
            nii=nibabel.Nifti1Image(FullSolutionPressureRefocus[mx[0]:mx[-1],my[0]:my[-1],mz[0]:mz[-1]],affine=affineSub)
            SaveNiftiEnforcedISO(nii,FILENAMES['FullElasticSolutionRefocus_Sub__'])
            ResaveNormalized(FILENAMES['FullElasticSolutionRefocus_Sub'],self._SkullMask)

                
        nii=nibabel.Nifti1Image(FullSolutionPressure[::ss,::ss,::ss],affine=affine)
        SaveNiftiEnforcedISO(nii,FILENAMES['FullElasticSolution__'])

        nii=nibabel.Nifti1Image(FullSolutionPhase[::ss,::ss,::ss],affine=affine)
        SaveNiftiEnforcedISO(nii,FILENAMES['FullElasticSolutionPhase__'])
        if bUseRayleighForWater:
            nii=nibabel.Nifti1Image(RayleighWater[::ss,::ss,::ss],affine=affine)
            SaveNiftiEnforcedISO(nii,FILENAMESWater['FullElasticSolution__'])
            nii=nibabel.Nifti1Image(RayleighWaterPhase[::ss,::ss,::ss],affine=affine)
            SaveNiftiEnforcedISO(nii,FILENAMESWater['FullElasticSolution__'])

        nii=nibabel.Nifti1Image(FullSolutionPressure[mx[0]:mx[-1],my[0]:my[-1],mz[0]:mz[-1]],affine=affineSub)
        SaveNiftiEnforcedISO(nii,FILENAMES['FullElasticSolution_Sub__'])
        ResaveNormalized(FILENAMES['FullElasticSolution_Sub'],self._SkullMask)

        if bUseRayleighForWater:
            nii=nibabel.Nifti1Image(RayleighWater[mx[0]:mx[-1],my[0]:my[-1],mz[0]:mz[-1]],affine=affineSub)
            SaveNiftiEnforcedISO(nii,FILENAMESWater['FullElasticSolution_Sub__'])
            ResaveNormalized(FILENAMESWater['FullElasticSolution_Sub'],self._SkullMask)

        if not bUseRayleighForWater: 
            nii=nibabel.Nifti1Image(RayleighWater[mx[0]:mx[-1],my[0]:my[-1],mz[0]:mz[-1]],affine=affineSub)
            SaveNiftiEnforcedISO(nii,FILENAMES['RayleighFreeWater__'].replace('RayleighFreeWater','RayleighFreeWater_Sub'))
        
        if subsamplingFactor>1:
            kt = ['p_amp','p_complex','MaterialMap']
            if 'MaterialMapCT' in DataForSim:
                kt.append('MaterialMapCT')
            if self._bDoRefocusing:
                kt.append('p_amp_refocus')
                kt.append('p_complex_refocus')
            if bUseRayleighForWater:
                kt.append('p_amp_water','p_complex_water')
            for k in kt:
                DataForSim[k]=DataForSim[k][::ss,::ss,::ss]
            for k in ['x_vec','y_vec','z_vec']:
                DataForSim[k]=DataForSim[k][::ss]
            DataForSim['SpatialStep']*=ss
            DataForSim['TargetLocation']=np.round(DataForSim['TargetLocation']/ss).astype(int)
        
        if bUseRayleighForWater:
            #We pop the water field temporarily
            p_amp_water =DataForSim.pop('p_amp_water')
            p_complex_water =DataForSim.pop('p_complex_water')

        DataForSim['bDoRefocusing']=self._bDoRefocusing
        DataForSim['affine']=affine

        DataForSim['TxMechanicalAdjustmentX']=self._TxMechanicalAdjustmentX
        DataForSim['TxMechanicalAdjustmentY']=self._TxMechanicalAdjustmentY
        DataForSim['TxMechanicalAdjustmentZ']=self._TxMechanicalAdjustmentZ
        DataForSim['ZIntoSkin']=self._ZIntoSkin
        DataForSim['ZIntoSkinPixels']=self._SIM_SETTINGS._ZIntoSkinPixels

        self.AddSaveDataSim(DataForSim)
        ###

        FocIJK=np.ones((4,1))
        DataMaskOrig=self._SkullMask.get_fdata()
        DataMask=np.flip(DataMaskOrig,axis=2)
        FocIJK[:3,0]=np.array(np.where(DataMask==5)).flatten()

        VoxelSize=self._SkullMask.header.get_zooms()[0]*1e-3
        LineOfSight=DataMask[int(FocIJK[0,0]),int(FocIJK[1,0]),:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (FocIJK[2,0]-StartSkin)*VoxelSize
        print('DistanceFromSkin',np.round(DistanceFromSkin*1e3,2))

        DataForSim['DistanceFromSkin']=DistanceFromSkin

        FocIJK[:3,0]=np.array(np.where(DataMaskOrig==5)).flatten()

        FocXYZ=self._SkullMask.affine@FocIJK
        FocIJKAdjust=FocIJK.copy()
        #we adjust in steps
        FocIJKAdjust[0,0]+=self._TxMechanicalAdjustmentX*1e3/self._SkullMask.header.get_zooms()[0]
        FocIJKAdjust[1,0]+=self._TxMechanicalAdjustmentY*1e3/self._SkullMask.header.get_zooms()[1]

        FocXYZAdjust=self._SkullMask.affine@FocIJKAdjust
        AdjustmentInRAS=(FocXYZ-FocXYZAdjust).flatten()[:3]
        DataForSim['AdjustmentInRAS']=AdjustmentInRAS
        print('Adjustment in RAS - T1W space',AdjustmentInRAS)
            
        sname=FILENAMES['DataForSim']
        if bMinimalSaving==False:
            SaveToH5py(DataForSim,sname)
            if bUseRayleighForWater:
                #we save now the h5 file for water
                DataForSim['p_amp']= p_amp_water
                DataForSim['p_complex']= p_complex_water
                if self._bDoRefocusing:
                    DataForSim.pop('p_amp_refocus')
                sname=FILENAMESWater['DataForSim']
                SaveToH5py(DataForSim,sname)

        gc.collect()
        
        return sname

    def OutPutConditions(self):
        ### Usage details

        String = 'Plese see below code implementing the complete simulation.\n'+\
                'Main highlights:\n\n'+\
                'Item  | value\n'+\
                '---- | ----\n'+\
                'PML size |  %i\n' %(self._SIM_SETTINGS._PMLThickness)+\
                'Spatial step$^*$ | $\\frac{\\lambda}{%i}$ = %3.2f mm (%i PPW)\n' %(self._SIM_SETTINGS._basePPW,\
                                                                                    self._SIM_SETTINGS._SpatialStep*1e3,\
                                                                                    self._SIM_SETTINGS._basePPW) +\
                'Final Interpolation at 0.5 mm | Linear for amplitude, nearest for phase\n'+\
                'FDTD solver | $O(2)$ temporal, $O(4)$ spatial, staggered grid\n'+\
                'Temporal step | %4.3f $\mu$s, %2.1f points-per-period\n'%(self._SIM_SETTINGS._TemporalStep*1e6,self._SIM_SETTINGS._PPP)+\
                'Adjusted CFL | %3.2f \n'%(self._SIM_SETTINGS._AdjustedCFL)+\
                'Source func. | CW-pulse for %4.1f $\mu$s\n' %(self._SIM_SETTINGS._TimeSimulation*1e6)+\
                'Amplitude method | Peak\n'+\
                'Phase method | NA in library, but it it is calculated from captured sensor data and FFT\n\n'+\
                '$^*$Spatial step chosen to produce peak pressure amplitude ~2% compared to reference simulation (FOCUS).'\
                
        return String
    
class SimulationConditionsBASE(object):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,baseMaterial=Material['Water'],
                      basePPW=9,
                      PMLThickness = 12, # grid points for perect matching layer, HIGHLY RECOMMENDED DO NOT CHANGE THIS SIZE 
                      ReflectionLimit= 1e-5, #DO NOT CHANGE THIS
                      DimDomain =  np.array([0.07,0.07,0.12]),
                      SensorSubSampling=6,
                      NumberCyclesToTrackAtEnd=2,
                      SourceAmp=60e3, # kPa
                      Frequency=500e3,
                      Aperture=0.16, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                      FocalLength=135e-3,
                      PaddingForKArray=0,
                      PaddingForRayleigh=0,
                      QfactorCorrection=True,
                      QCorrection=3,
                      bDisplay=True,
                      bTightNarrowBeamDomain = False,
                      zLengthBeyonFocalPointWhenNarrow=4e-2,
                      TxMechanicalAdjustmentX =0, # in case we want to move mechanically the Tx (useful when targeting shallow locations such as M1 and we want to evaluate if an small mechanical adjustment can ensure focusing)
                      TxMechanicalAdjustmentY =0, # in case we want to move mechanically the Tx (useful when targeting shallow locations such as M1 and we want to evaluate if an small mechanical adjustment can ensure focusing)
                      TxMechanicalAdjustmentZ =0, # in case we want to move mechanically the Tx (useful when targeting shallow locations such as M1 and we want to evaluate if an small mechanical adjustment can ensure focusing)
                      ZIntoSkin=0.0, # in case we want to push the Tx "into" the skin simulating compressing the Tx in the scalp (removing tissue layers)
                      ZTxCorrecton=0.0, # this compensates for flat transducers that have a dead space before reaching the skin
                      DensityCTMap=None, #use CT map
                      ExtraDepthAdjust= 0.0, #for any need to stretch the cone used to calculate the cross section are
                      ExtraAdjustX =[0.0],
                      ExtraAdjustY =[0.0],
                      bSaveStress=False,
                      bSaveDisplacement=False,
                      DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721],#coefficients to correct for values lower of CFL =1.0 in water conditions.
                      BenchmarkTestFile='',
                      InputFocusStart='',
                      OptimizedWeightsFile=''): #file with optimized weights for the Tx  
        self._Materials=[[baseMaterial[0],baseMaterial[1],baseMaterial[2],baseMaterial[3],baseMaterial[4]]]
        self._basePPW=basePPW
        self._PMLThickness=PMLThickness
        self._ReflectionLimit=ReflectionLimit
        self._ODimDomain =DimDomain 
        self._SensorSubSampling=SensorSubSampling
        self._NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd
        self._TemporalStep=0.
        self._N1=0
        self._N2=0
        self._N3=0
        self._FactorConvPtoU=baseMaterial[0]*baseMaterial[1]
        self._SourceAmpPa=SourceAmp
        self._SourceAmpDisplacement=SourceAmp/self._FactorConvPtoU
        self._Frequency=Frequency
        self._weight_amplitudes=1.0
        self._PaddingForKArray=PaddingForKArray
        self._PaddingForRayleigh=PaddingForRayleigh
        self._QfactorCorrection=QfactorCorrection
        self._QCorrection=QCorrection
        self._bDisplay=bDisplay
        self._DispersionCorrection=DispersionCorrection
        self._Aperture=Aperture
        self._FocalLength=FocalLength
        self._bTightNarrowBeamDomain=bTightNarrowBeamDomain
        self._zLengthBeyonFocalPointWhenNarrow=zLengthBeyonFocalPointWhenNarrow
        self._TxMechanicalAdjustmentX=TxMechanicalAdjustmentX
        self._TxMechanicalAdjustmentY=TxMechanicalAdjustmentY
        self._TxMechanicalAdjustmentZ=TxMechanicalAdjustmentZ
        self._ZIntoSkin=ZIntoSkin
        self._DensityCTMap=DensityCTMap
        self._ZIntoSkinPixels=0 # To be updated in UpdateConditions
        self._ZSourceLocation= 0.0 # To be updated in UpdateConditions
        self._ExtraDepthAdjust=ExtraDepthAdjust
        self._ExtraAdjustX =ExtraAdjustX
        self._ExtraAdjustY =ExtraAdjustY
        self._ZTxCorrecton=ZTxCorrecton
        self._bSaveStress=bSaveStress
        self._bSaveDisplacement=bSaveDisplacement
        self._BenchmarkTestFile=BenchmarkTestFile
        self._InputFocusStart=InputFocusStart
        self._OptimizedWeightsFile=''
        self._OptimizedWeights=None
        if len(OptimizedWeightsFile)>0 :
            print('Simulation Using OptimizedWeightsFile',OptimizedWeightsFile)
            if not os.path.isfile(OptimizedWeightsFile):
                raise FileNotFoundError("OptimizedWeightsFile %s does not exist." %(OptimizedWeightsFile))
            self._OptimizedWeightsFile=OptimizedWeightsFile
            self._OptimizedWeights = np.load(OptimizedWeightsFile,allow_pickle=True)['res'].flatten()[0].x
 
    def AddMaterial(self,Density,LSoS,SSoS,LAtt,SAtt): #add material (Density (kg/m3), long. SoS 9(m/s), shear SoS (m/s), Long. Attenuation (Np/m), shear attenuation (Np/m)
        self._Materials.append([Density,LSoS,SSoS,LAtt,SAtt]);
        
    def ResetMaterial(self): #reset material list
        self._Materials=[]
        
        
    @property
    def Wavelength(self):
        return self._Wavelength
        
        
    @property
    def SpatialStep(self):
        return self._SpatialStep
        
    def UpdateConditions(self, SkullMaskNii,AlphaCFL=1.0,bWaterOnly=False,
                         bForceHomogenousMedium=False,
                         BenchmarkTestFile=''):
        '''
        Update simulation conditions
        '''
        MatArray=self.ReturnArrayMaterial()
        SmallestSOS=np.sort(MatArray[:,1:3].flatten())
        iS=np.where(SmallestSOS>0)[0]
        if len(BenchmarkTestFile)==0:
            SmallestSOS=np.min([SmallestSOS[iS[0]],GetSmallestSOS(self._Frequency,bShear=True)])
        else:
            InputDataBenchmark=ReadFromH5py(BenchmarkTestFile)
            SmallestSOS=1e6
            for e in InputDataBenchmark['Materials']:
                SmallestSOS=np.min((SmallestSOS,e['LongSoS']))
                if e['ShearSoS']!=0:
                    SmallestSOS=np.min((SmallestSOS,e['ShearSoS']))
        self._Wavelength=SmallestSOS/self._Frequency
        self._baseAlphaCFL =AlphaCFL
        print(" Wavelength, baseAlphaCFL",self._Wavelength,AlphaCFL)
        print ("smallSOS ", SmallestSOS)
        
        SpatialStep=self._Wavelength/self._basePPW
        
        dummyMaterialMap=np.zeros((10,10,MatArray.shape[0]),np.uint32)
        for n in range(MatArray.shape[0]):
            dummyMaterialMap[:,:,n]=n
        
        OTemporalStep,_,_, _, _,_,_,_,_,_=PModel.CalculateMatricesForPropagation(dummyMaterialMap,MatArray,self._Frequency,self._QfactorCorrection,SpatialStep,AlphaCFL)
        
        self.DominantMediumTemporalStep,_,_, _, _,_,_,_,_,_=PModel.CalculateMatricesForPropagation(dummyMaterialMap*0,MatArray[0,:].reshape((1,5)),self._Frequency,self._QfactorCorrection,SpatialStep,1.0)

        TemporalStep=OTemporalStep

        print('"ideal" TemporalStep',TemporalStep)
        print('"ideal" DominantMediumTemporalStep',self.DominantMediumTemporalStep)

        #now we make it to be an integer division of the period
        self._PPP=np.ceil(1/self._Frequency/TemporalStep)
        #we add to catch the weird case it ends in large prime number, to avoid having a sensor that needs so many points
        if self._PPP==31:
            self._PPP=32
        elif self._PPP==34:
            self._PPP=35
        elif self._PPP==23:
            self._PPP=24
        elif self._PPP==71:
            self._PPP=72
        elif self._PPP==74:
            self._PPP=75
        elif self._PPP==79:
            self._PPP=80
        elif self._PPP==47:
            self._PPP=48
        elif self._PPP %5 !=0:
            self._PPP=(self._PPP//5 +1)*5

        TemporalStep=1/self._Frequency/self._PPP # we make it an integer of the period
        self._AdjustedCFL=TemporalStep/OTemporalStep*AlphaCFL
        
        #and back to SpatialStep
        print('"int fraction" TemporalStep',TemporalStep)
        print('"CFL fraction relative to water only conditions',TemporalStep/self.DominantMediumTemporalStep)
        
        print("adjusted AlphaCL, PPP",self._AdjustedCFL,self._PPP)
        
        self._SpatialStep=SpatialStep
        self._TemporalStep=TemporalStep

        self._ZIntoSkinPixels=int(np.round(self._ZIntoSkin/SpatialStep))
        self._ZSourceLocation=self._ZIntoSkinPixels+self._PMLThickness
        
        #we save the mask array and flipped
        self._SkullMaskDataOrig=np.flip(SkullMaskNii.get_fdata(),axis=2)
        self._SkullMaskNii=SkullMaskNii
        voxelS=np.array(SkullMaskNii.header.get_zooms())*1e-3
        print('voxelS, SpatialStep',voxelS,SpatialStep)
        if not (np.allclose(np.round(np.ones(voxelS.shape)*SpatialStep,6),np.round(voxelS,6))):
            print('*'*40)
            print('Warning: voxel size in input Nifti and the expected size not identical',voxelS,SpatialStep)
            print('*'*40)
        
        self._XLOffset=self._PMLThickness 
        self._YLOffset=self._PMLThickness
        

        #default offsets , this can change if the Rayleigh field does not fit
        self._ZLOffset=self._PMLThickness+self._PaddingForRayleigh+self._PaddingForKArray
        self._ZLOffset+=int(np.round(self._ZTxCorrecton/self._SpatialStep))
        self._XROffset=self._PMLThickness 
        self._YROffset=self._PMLThickness
        self._ZROffset=self._PMLThickness
        
        
        #swe will adjust size of domain until be sure the incident Rayleigh field fits in
        self._XShrink_L=0
        self._XShrink_R=0
        self._YShrink_L=0
        self._YShrink_R=0
        self._ZShrink_L=0
        self._ZShrink_R=0
        self.bMapFit=False
        bCompleteForShrinking=False
        self._nCountShrink=0
        print('self._ExtraAdjustX, self._ExtraAdjustY',self._ExtraAdjustX,self._ExtraAdjustY)
        while not self.bMapFit or not bCompleteForShrinking:
            self.bMapFit=True
            self._N1=self._SkullMaskDataOrig.shape[0]+self._XLOffset+self._XROffset -self._XShrink_L-self._XShrink_R
            self._N2=self._SkullMaskDataOrig.shape[1]+self._YLOffset+self._YROffset -self._YShrink_L-self._YShrink_R
            self._N3=self._SkullMaskDataOrig.shape[2]+self._ZLOffset+self._ZROffset -self._ZShrink_L-self._ZShrink_R 


            self._FocalSpotLocationOrig=np.array(np.where(self._SkullMaskDataOrig==5.0)).flatten()
            self._FocalSpotLocation=self._FocalSpotLocationOrig.copy()
            self._FocalSpotLocation+=np.array([self._XLOffset,self._YLOffset,self._ZLOffset])
            self._FocalSpotLocation-=np.array([self._XShrink_L,self._YShrink_L,self._ZShrink_L])
            print('self._FocalSpotLocation',self._FocalSpotLocation)
            
            xfield = np.arange(self._N1)*SpatialStep
            yfield = np.arange(self._N2)*SpatialStep
            zfield = np.arange(self._N3)*SpatialStep
            
            
            xfield-=xfield[self._FocalSpotLocation[0]]
            yfield-=yfield[self._FocalSpotLocation[1]]
            zfield-=zfield[self._FocalSpotLocation[2]]
            
            zfield+=self._FocalLength
            TopZ=zfield[self._PMLThickness]
            
            if self._FocalLength!=0:
                DistanceToFocus=self._FocalLength-TopZ+self._TxMechanicalAdjustmentZ+self._ExtraDepthAdjust
                Alpha=np.arcsin(self._Aperture/2/(self._FocalLength+self._ExtraDepthAdjust))
                RadiusFace=DistanceToFocus*np.tan(Alpha)*1.10 # we make a bit larger to be sure of covering all incident beam
            else:
                RadiusFace=self._Aperture/2*1.10

            if len(BenchmarkTestFile)>0:
            #we adjust dimensions to benchmark 1
                print('Forcing radiusface to fit benchmark 1, current value', RadiusFace)
                if RadiusFace>0.035:
                    warnings.warn('RadiusFace too large, setting to 35 mm')
                RadiusFace=0.035
            
            print('RadiusFace',RadiusFace)
            print('yfield',yfield.min(),yfield.max())
            
            ypp,xpp=np.meshgrid(yfield,xfield)
            
            RegionMap=((xpp-self._TxMechanicalAdjustmentX)**2+(ypp-self._TxMechanicalAdjustmentY)**2)<=RadiusFace**2 #we select the circle on the incident field
            for EX,EY in zip (self._ExtraAdjustX,self._ExtraAdjustY):
                RegionMap=(RegionMap)|(((xpp-self._TxMechanicalAdjustmentX-EX)**2+(ypp-self._TxMechanicalAdjustmentY-EY)**2)<=RadiusFace**2)
                IndXMap,IndYMap=np.nonzero(RegionMap)
            print('RegionMap',np.sum(RegionMap))
            
            def fgen(var):
                sn={'X':'1','Y':'2','Z':'3'}
                pcode=\
'''

if np.any(Ind{0}Map<self._PMLThickness):
    print('** Rayleigh map not fitting in the low part of N{1}, increasing it ...', self._{0}LOffset)
    self._{0}LOffset+=self._PMLThickness-Ind{0}Map.min()
    print('{0}LOffset',self._{0}LOffset)
    self.bMapFit=False
elif self._bTightNarrowBeamDomain:
    if self._{0}LOffset==self._PMLThickness:
        self._{0}Shrink_L+=Ind{0}Map.min()-self._{0}LOffset
        print('{0}Shrink_L',self._{0}Shrink_L)
    self._nCountShrink+=1
if np.any(Ind{0}Map>=self._N{1}-self._PMLThickness):
    print('** Rayleigh map not fitting in the upper part of N{1}, increasing it ...',self._{0}ROffset)
    self._{0}ROffset+=Ind{0}Map.max()-(self._N{1}-self._PMLThickness)+1
    print('{0}Offset',self._{0}ROffset)
    self.bMapFit=False
elif self._bTightNarrowBeamDomain:
    if self._{0}ROffset==self._PMLThickness:
        self._{0}Shrink_R+=self._N{1}-self._{0}ROffset-Ind{0}Map.max()-1
        print('{0}Shrink_R',self._{0}Shrink_R)
    self._nCountShrink+=1
'''.format(var,sn[var])
                return pcode
            
            
            exec(fgen('X'))
            exec(fgen('Y'))

            if self._bTightNarrowBeamDomain:
                nStepsZReduction=int(self._zLengthBeyonFocalPointWhenNarrow/self._SpatialStep)
                self._ZShrink_R+=self._N3-(self._FocalSpotLocation[2]+nStepsZReduction)
                if self._ZShrink_R<0:
                    self._ZShrink_R=0
                print('ZShrink_R',self._ZShrink_R,self._nCountShrink)
                    
            if self.bMapFit:
                if self._bTightNarrowBeamDomain==False:
                    bCompleteForShrinking=True
                elif self._nCountShrink>=8:
                    bCompleteForShrinking=True

            #we overwrite the values if benchmark is being selected
            
        self._XDim=xfield
        self._YDim=yfield
        self._ZDim=zfield
        
        
        print('Domain size',self._N1,self._N2,self._N3)
        self._DimDomain=np.zeros((3))
        self._DimDomain[0]=self._N1*SpatialStep
        self._DimDomain[1]=self._N2*SpatialStep
        self._DimDomain[2]=self._N3*SpatialStep
        
        self._TimeSimulation=np.sqrt((self._DimDomain[0]-self._PMLThickness*2*SpatialStep)**2+
                                     (self._DimDomain[1]-self._PMLThickness*2*SpatialStep)**2+
                                     (self._DimDomain[2]-self._PMLThickness*2*SpatialStep)**2)/MatArray[0,1] #time to cross one corner to another
        self._TimeSimulation=np.floor(self._TimeSimulation/self._TemporalStep)*self._TemporalStep
        
        TimeVector=np.arange(0.0,self._TimeSimulation,self._TemporalStep)
        ntSteps=(int(TimeVector.shape[0]/self._PPP)+1)*self._PPP
        self._TimeSimulation=self._TemporalStep*ntSteps
        TimeVector=np.arange(0,ntSteps)*self._TemporalStep
        bRecalcSubSampling=False
        if self._SensorSubSampling==0:
            bRecalcSubSampling=True
        elif self._PPP % self._SensorSubSampling !=0:
            bRecalcSubSampling=True
        if bRecalcSubSampling:
            print('overwrriting  self._SensorSubSampling')
            potential=np.arange(1,self._PPP).tolist()
            result = np.array(list(filter(lambda x: (self._PPP % x == 0), potential)))
            AllDiv=self._PPP/result
            result=result[AllDiv>=4]
            AllDiv=self._PPP/result
            self._SensorSubSampling=int(result[-1])
            assert(AllDiv[-1]<=10) 

        print('PPP, Subsampling, PPP for sensors ', self._PPP,self._SensorSubSampling, self._PPP/self._SensorSubSampling)
        assert((self._PPP%self._SensorSubSampling)==0)
        nStepsBack=int(self._NumberCyclesToTrackAtEnd*self._PPP)
        self._SensorStart=int((TimeVector.shape[0]-nStepsBack)/self._SensorSubSampling)

        self._MaterialMap=np.zeros((self._N1,self._N2,self._N3),np.uint32) # note the 32 bit size
        if bWaterOnly==False and bForceHomogenousMedium == False and len(BenchmarkTestFile)==0:
            #we add the material map
            if self._XShrink_R==0:
                upperXR=self._SkullMaskDataOrig.shape[0]
            else:
                upperXR=-self._XShrink_R
            if self._YShrink_R==0:
                upperYR=self._SkullMaskDataOrig.shape[1]
            else:
                upperYR=-self._YShrink_R
            if self._ZShrink_R==0:
                upperZR=self._SkullMaskDataOrig.shape[2]
            else:
                upperZR=-self._ZShrink_R
                
            self._MaterialMap[self._XLOffset:-self._XROffset,
                              self._YLOffset:-self._YROffset,
                              self._ZLOffset:-self._ZROffset]=\
                                self._SkullMaskDataOrig.astype(np.uint32)[self._XShrink_L:upperXR,
                                                                         self._YShrink_L:upperYR,
                                                                         self._ZShrink_L:upperZR]
            bBrainSegmentation = np.any(self._MaterialMap>5)
            if self._DensityCTMap is not None:
                assert(self._DensityCTMap.dtype==np.uint32)
                BoneRegion=(self._MaterialMap==2) | (self._MaterialMap==3)
                self._MaterialMapNoCT=self._MaterialMap.copy()
                if bBrainSegmentation:
                    #we re arrange labels
                    self._MaterialMap[self._MaterialMap==4]=2 
                    self._MaterialMap[self._MaterialMap==5]=2 #we define target as regular brain (we wil need to fix this later)
                    self._MaterialMap[self._MaterialMap>=6]-=3   
                else:
                    self._MaterialMap[self._MaterialMap>=4]=2 # Brain region is in material 2
                SubCTMap=np.zeros_like(self._MaterialMap)
                SubCTMap[self._XLOffset:-self._XROffset,
                              self._YLOffset:-self._YROffset,
                              self._ZLOffset:-self._ZROffset]=\
                                self._DensityCTMap[self._XShrink_L:upperXR,
                                                                         self._YShrink_L:upperYR,
                                                                         self._ZShrink_L:upperZR]
                self._MaterialMap[BoneRegion]=SubCTMap[BoneRegion]
                assert(SubCTMap[BoneRegion].min()>=3)
                assert(SubCTMap[BoneRegion].max()<=self.ReturnArrayMaterial().shape[0])

            else:
                if bBrainSegmentation:
                    self._MaterialMap[self._MaterialMap>=5]-=1 
                else:
                    self._MaterialMap[self._MaterialMap==5]=4 # this is to make the focal spot location as brain tissue

            #We remove tissue layers
            self._MaterialMap[:,:,:self._ZSourceLocation+1] = 0 # we remove tissue layers by putting water
        
        #####
        ##### bForceHomogenousMedium and BenchmarkTestFile are only for testing
        #####
        if bForceHomogenousMedium:
            self._MaterialMap[:,:,:]=1
        if len(BenchmarkTestFile)>0 and not bWaterOnly:
            assert(np.all(np.array(self._MaterialMap.shape)==np.array(InputDataBenchmark['MaterialMap'].shape)))
            assert(self._MaterialMap.dtype==InputDataBenchmark['MaterialMap'].dtype)
            
            self._MaterialMap=InputDataBenchmark['MaterialMap']
            
        print('PPP, Duration simulation',np.round(1/self._Frequency/TemporalStep),self._TimeSimulation*1e6)
        
        print('Number of steps sensor',np.floor(self._TimeSimulation/self._TemporalStep/self._SensorSubSampling)-self._SensorStart)
        
        
    def AdjustWeightAmplitudes(self):
        '''
        Adjust the weight amplitudes to be used in the simulation
        '''
        if self._OptimizedWeights is not None:
            print('Using optimized weights from file %s' %(self._OptimizedWeightsFile))
            if self._OptimizedWeights.size != self._TxRC['center'].shape[0]:
                raise ValueError("OptimizedWeightsFile %s has %i subelements, but the Tx has %i elements." %(self._OptimizedWeightsFile,self._OptimizedWeights.size,self._TxRC['center'].shape[0]))
            return self._OptimizedWeights.reshape((self._TxRC['center'].shape[0],1)).astype(np.complex64)
        else:
            return 1.0

    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        raise NotImplementedError("Need to implement this")
           
    def ReturnArrayMaterial(self):
        return np.array(self._Materials)

    def CreateSources(self,ramp_length=4):
        raise NotImplementedError("Need to implement this")
 
    def CreateSensorMap(self):
        
        self._SensorMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        # for the back propagation, we only use the entering face
        self._SensorMapBackPropagation=np.zeros((self._N1,self._N2,self._N3),np.uint32)    
    
        self._SensorMapBackPropagation[self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness,self._PMLThickness]=1
        self._SensorMap[self._PMLThickness:-self._PMLThickness,
                        self._PMLThickness:-self._PMLThickness,
                        self._ZSourceLocation+1:-self._PMLThickness]=1
              
        if self._bDisplay:
            plt.figure()
            plt.imshow(self._SensorMap[:,self._FocalSpotLocation[1],:].T,cmap=plt.cm.gray)
            plt.title('Sensor map location')
        
            
        
    def RUN_SIMULATION(self,GPUName='SUPER',SelMapsRMSPeakList=['Pressure'],bRefocused=False,
                       bApplyCorrectionForDispersion=True,
                       COMPUTING_BACKEND=1,bDoRefocusing=True):
        MaterialList=self.ReturnArrayMaterial()

        TypeSource=0 #particle source
        Ox=np.zeros(self._MaterialMap.shape) 
        Oy=np.zeros(self._MaterialMap.shape) 
        Oz=np.ones(self._MaterialMap.shape)/self._FactorConvPtoU

        if bRefocused==False:
            self._Sensor,LastMap,self._DictPeakValue,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                             self._MaterialMap,
                                                             MaterialList,
                                                             self._Frequency,
                                                             self._SourceMap,
                                                             self._PulseSource,
                                                             self._SpatialStep,
                                                             self._TimeSimulation,
                                                             self._SensorMap,
                                                             Ox=Ox,
                                                             Oy=Oy,
                                                             Oz=Oz,
                                                             NDelta=self._PMLThickness,
                                                             DT=self._TemporalStep,
                                                             ReflectionLimit=self._ReflectionLimit,
                                                             COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                             USE_SINGLE=True,
                                                             SelMapsRMSPeakList=SelMapsRMSPeakList,
                                                             SelMapsSensorsList=['Pressure'],
                                                             SelRMSorPeak=1,
                                                             DefaultGPUDeviceName=GPUName,
                                                             AlphaCFL=1.0,
                                                             TypeSource=TypeSource,
                                                             QfactorCorrection=self._QfactorCorrection,
                                                             QCorrection=self._QCorrection,
                                                             SensorSubSampling=self._SensorSubSampling,
                                                             SensorStart=self._SensorStart)
            
            self._InputParam=InputParam['IndexSensorMap']
            gc.collect()

            if bDoRefocusing:
            #now backpropagation
                self._SensorBack,_,_,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                                 self._MaterialMap,
                                                                 MaterialList,
                                                                 self._Frequency,
                                                                 self._SourceMapPunctual,
                                                                 self._PunctualSource,
                                                                 self._SpatialStep,
                                                                 self._TimeSimulation,
                                                                 self._SensorMapBackPropagation,
                                                                 Ox=Ox,
                                                                 Oy=Oy,
                                                                 Oz=Oz,
                                                                 NDelta=self._PMLThickness,
                                                                 DT=self._TemporalStep,
                                                                 ReflectionLimit=self._ReflectionLimit,
                                                                 COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                                 USE_SINGLE=True,
                                                                 SelMapsRMSPeakList=['Pressure'],
                                                                 SelMapsSensorsList=['Pressure'],
                                                                 SelRMSorPeak=1,
                                                                 DefaultGPUDeviceName=GPUName,
                                                                 AlphaCFL=1.0,
                                                                 TypeSource=TypeSource,
                                                                 QfactorCorrection=self._QfactorCorrection,
                                                                 QCorrection=self._QCorrection,
                                                                 SensorSubSampling=self._SensorSubSampling,
                                                                 SensorStart=self._SensorStart)
                self._InputParamBack=InputParam['IndexSensorMap']
        else:
            self._SensorRefocus,_,self._DictPeakValueRefocus,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                             self._MaterialMap,
                                                             MaterialList,
                                                             self._Frequency,
                                                             self._SourceMap,
                                                             self._PulseSourceRefocus,
                                                             self._SpatialStep,
                                                             self._TimeSimulation,
                                                             self._SensorMap,
                                                             Ox=Ox,
                                                             Oy=Oy,
                                                             Oz=Oz,
                                                             NDelta=self._PMLThickness,
                                                             DT=self._TemporalStep,
                                                             ReflectionLimit=self._ReflectionLimit,
                                                             COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                             USE_SINGLE=True,
                                                             SelMapsRMSPeakList=SelMapsRMSPeakList,
                                                             SelMapsSensorsList=['Pressure'],
                                                             SelRMSorPeak=1,
                                                             DefaultGPUDeviceName=GPUName,
                                                             AlphaCFL=1.0,
                                                             TypeSource=TypeSource,
                                                             QfactorCorrection=self._QfactorCorrection,
                                                             QCorrection=self._QCorrection,
                                                             SensorSubSampling=self._SensorSubSampling,
                                                             SensorStart=self._SensorStart)
            self._InputParamRefocus=InputParam['IndexSensorMap']

        print('self._DictPeakValue keys',self._DictPeakValue.keys())

        if bApplyCorrectionForDispersion:
            CFLWater=self._TemporalStep/self.DominantMediumTemporalStep
            ExpectedError=np.polyval(self._DispersionCorrection,CFLWater)
            Correction=100.0/(100.0-ExpectedError)
            print('CFLWater only, ExpectedError, Correction', CFLWater,ExpectedError,Correction)
            if bRefocused==False:
                for k in self._DictPeakValue:
                    self._DictPeakValue[k]*=Correction*np.sqrt(2)
                for k in self._Sensor:
                    if k=='time':
                        continue
                    self._Sensor[k]*=Correction
                if bDoRefocusing:
                    for k in self._SensorBack:
                        if k=='time':
                            continue
                        self._SensorBack[k]*=Correction
            else:
                for k in self._DictPeakValueRefocus:
                    self._DictPeakValueRefocus[k]*=Correction*np.sqrt(2)
                for k in self._SensorRefocus:
                    if k=='time':
                        continue
                    self._SensorRefocus[k]*=Correction
                    
        gc.collect()
    
    def CalculatePhaseData(self,bRefocused=False,bDoRefocusing=True):
        
        t0=time.time()
        if bRefocused==False:
            self._PhaseMap=np.zeros((self._N1,self._N2,self._N3),np.float32)
            self._PressMapFourier=np.zeros((self._N1,self._N2,self._N3),np.complex64)
            self._PressMapPeak=np.zeros((self._N1,self._N2,self._N3),np.float32)
            if bDoRefocusing:
                self._PressMapFourierBack=np.zeros((self._N1,self._N2),np.complex64)
        else:
            self._PhaseMapRefocus=np.zeros((self._N1,self._N2,self._N3),np.float32)
            self._PressMapFourierRefocus=np.zeros((self._N1,self._N2,self._N3),np.complex64)
            self._PressMapPeakRefocus=np.zeros((self._N1,self._N2,self._N3),np.float32)
            
   
        time_step = np.diff(self._Sensor['time']).mean() #remember the sensor time vector can be different from the input source
        
        if self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling) !=0: #because some roundings, we may get
            print('Rounding of time vector was not exact multiple of PPP, truncating time vector a little')
            nDiff=int(self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling))
            print(' Cutting %i entries from sensor from length %i to %i' %(nDiff,self._Sensor['time'].shape[0],self._Sensor['time'].shape[0]-nDiff))
            self._Sensor['time']=self._Sensor['time'][:-nDiff]
            self._Sensor['Pressure']=self._Sensor['Pressure'][:-nDiff]
        assert((self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling))==0)

        freqs = np.fft.fftfreq(self._Sensor['time'].size, time_step)
        IndSpectrum=np.argmin(np.abs(freqs-self._Frequency)) # frequency entry closest to 500 kHz
        if bRefocused==False:
            self._Sensor['Pressure']=np.ascontiguousarray(self._Sensor['Pressure'])
            
            index=self._InputParam-1
            nStep=100000
            for n in range(0,self._Sensor['Pressure'].shape[0],nStep):
                top=np.min([n+nStep,self._Sensor['Pressure'].shape[0]])
                FSignal=fft.fft(self._Sensor['Pressure'][n:top,:],axis=1)
                k=index[n:top]//(self._N1*self._N2)
                j=index[n:top]%(self._N1*self._N2)
                i=j%self._N1
                j=j//self._N1
                FSignal=FSignal[:,IndSpectrum]
                pa= np.angle(FSignal)
                pp=np.abs(FSignal)

                self._PhaseMap[i,j,k]=pa
                self._PressMapFourier[i,j,k]=FSignal
                self._PressMapPeak[i,j,k]=self._Sensor['Pressure'][n:top,:].max(axis=1)
            self._InPeakValue=self._DictPeakValue['Pressure']
            self._PressMapFourier*=2/self._Sensor['time'].size
            print('Elapsed time doing phase and amp extraction from Fourier (s)',time.time()-t0)
            
            if bDoRefocusing:
                self._SensorBack['Pressure']=np.ascontiguousarray(self._SensorBack['Pressure'])
                index=self._InputParamBack-1
                for n in range(0,self._SensorBack['Pressure'].shape[0],nStep):
                    top=np.min([n+nStep,self._SensorBack['Pressure'].shape[0]])
                    FSignal=fft.fft(self._SensorBack['Pressure'][n:top,:],axis=1)
                    k=index[n:top]//(self._N1*self._N2)
                    j=index[n:top]%(self._N1*self._N2)
                    i=j%self._N1
                    j=j//self._N1
                    FSignal=FSignal[:,IndSpectrum]
                    assert(np.all(k==self._PMLThickness))
                    self._PressMapFourierBack[i,j]=FSignal
                    
        else:
            self._SensorRefocus['Pressure']=np.ascontiguousarray(self._SensorRefocus['Pressure'])
            index=self._InputParamRefocus-1
            nStep=100000
            for n in range(0,self._SensorRefocus['Pressure'].shape[0],nStep):
                top=np.min([n+nStep,self._SensorRefocus['Pressure'].shape[0]])
                FSignal=fft.fft(self._SensorRefocus['Pressure'][n:top,:],axis=1)
                k=index[n:top]//(self._N1*self._N2)
                j=index[n:top]%(self._N1*self._N2)
                i=j%self._N1
                j=j//self._N1
                FSignal=FSignal[:,IndSpectrum]
                pa= np.angle(FSignal)
                pp=np.abs(FSignal)

                self._PhaseMapRefocus[i,j,k]=pa
                self._PressMapFourierRefocus[i,j,k]=FSignal
                self._PressMapPeakRefocus[i,j,k]=self._SensorRefocus['Pressure'][n:top,:].max(axis=1)
            self._InPeakValueRefocus=self._DictPeakValueRefocus['Pressure']
            self._PressMapFourierRefocus*=2/self._SensorRefocus['time'].size
            print('Elapsed time doing phase and amp extraction from Fourier (s)',time.time()-t0)
            
        
            
    def BackPropagationRayleigh(self,deviceName='6800'):
        raise NotImplementedError("Need to implement this") 
        
    def CreateSourcesRefocus(self,ramp_length=4):
        raise NotImplementedError("Need to implement this")
        
    def PlotResultsPlanePartial(self):
  
        if self._bDisplay:
            plt.figure(figsize=(18,6))
            plt.subplot(1,3,1)
            plt.imshow(self._PressMapPeak[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD peak amp. (MPa)')
            plt.subplot(1,3,2)
            plt.imshow(self._PressMapFourier[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD Fourier amp. (MPa)')
            plt.subplot(1,3,3)
            plt.imshow(self._InPeakValue[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD InPeak amp. (MPa)')
            
    def PlotResultsPlane(self,bDoRefocusing=True):
  
        if self._bDisplay:
            if bDoRefocusing:
                plt.figure(figsize=(18,12))
                nRows=2
            else:
                plt.figure(figsize=(18,7))
                nRows=1
            plt.subplot(nRows,3,1)
            plt.imshow(self._PressMapPeak[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD peak amp. (MPa)')
            plt.subplot(nRows,3,2)
            plt.imshow(self._PressMapFourier[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD Fourier amp. (MPa)')
            plt.subplot(nRows,3,3)
            plt.imshow(self._InPeakValue[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._XDim.min(),self._XDim.max())
            plt.ylim(self._ZDim.max(),self._ZDim.min())
            plt.colorbar()
            plt.title('BabelViscoFDTD InPeak amp. (MPa)')
            
            if bDoRefocusing:
                plt.subplot(nRows,3,4)
                plt.imshow(self._PressMapPeakRefocus[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
                plt.xlim(self._XDim.min(),self._XDim.max())
                plt.ylim(self._ZDim.max(),self._ZDim.min())
                plt.colorbar()
                plt.title('BabelViscoFDTD peak refocus (MPa)')
                plt.subplot(nRows,3,5)
                plt.imshow(self._PressMapFourierRefocus[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
                plt.xlim(self._XDim.min(),self._XDim.max())
                plt.ylim(self._ZDim.max(),self._ZDim.min())
                plt.colorbar()
                plt.title('BabelViscoFDTD Fourier Refocus (MPa)')
                plt.subplot(nRows,3,6)
                plt.imshow(self._InPeakValueRefocus[:,self._FocalSpotLocation[1],:].T/1e6,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
                plt.xlim(self._XDim.min(),self._XDim.max())
                plt.ylim(self._ZDim.max(),self._ZDim.min())
                plt.colorbar()
                plt.title('BabelViscoFDTD InPeak amp. Refocus (MPa)')
            


        LineInPeak=self._InPeakValue[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]/1e6
        LineFourierAmp=self._PressMapFourier[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]/1e6
        if bDoRefocusing:
            LineInPeakRefocus=self._InPeakValueRefocus[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]/1e6
            LinePeakRefocus=self._PressMapPeakRefocus[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]/1e6
            LineFourierAmpRefocus=self._PressMapFourierRefocus[self._FocalSpotLocation[0],self._FocalSpotLocation[1],:]/1e6

        if self._bDisplay:
            Z=self._ZDim*1e3
            fig, ax = plt.subplots(1,1,figsize=(12,8))
            ax.plot(Z,LineInPeak)
            ax.plot(Z,LineFourierAmp)
            if bDoRefocusing:
                ax.plot(Z,LinePeakRefocus)
                ax.plot(Z,LineInPeakRefocus)
                ax.plot(Z,LineFourierAmpRefocus)
            if bDoRefocusing:
                plt.legend(['Inpeak','Fourier','PeakRefocus','InpeakRefocus','FourierRefocus'])
            else:
                plt.legend(['Inpeak','Fourier'])
            ax.plot([self._FocalLength*1e3,self._FocalLength*1e3],
                     [0,np.max(LineInPeak)],':')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        
    def ReturnResults(self,bDoRefocusing=True,bUseRayleighForWater=False):
        if self._XShrink_R==0:
            upperXR=self._SkullMaskDataOrig.shape[0]
        else:
            upperXR=-self._XShrink_R
        if self._YShrink_R==0:
            upperYR=self._SkullMaskDataOrig.shape[1]
        else:
            upperYR=-self._YShrink_R
        if self._ZShrink_R==0:
            upperZR=self._SkullMaskDataOrig.shape[2]
        else:
            upperZR=-self._ZShrink_R

        self._u2RayleighField[:,:,:self._ZSourceLocation+1]=0.0
        
        #we return the region not including the PML and padding
        RayleighWater=np.zeros(self._SkullMaskDataOrig.shape,np.complex64)
        RayleighWater[self._XShrink_L:upperXR,
                      self._YShrink_L:upperYR,
                      self._ZShrink_L:upperZR]=\
                      self._u2RayleighField[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset]
        
        RayleighWater=np.flip(RayleighWater,axis=2)
        RayleighWaterPhase=np.angle(RayleighWater)
        RayleighWater=np.abs(RayleighWater)
        
        #this one creates an overlay of skull and brain tissue that helps to show it Slicer or other visualization tools
        MaskSkull=np.flip(self._SkullMaskDataOrig.astype(np.float32),axis=2)
        MaskCalcRegions=np.zeros(MaskSkull.shape,bool)
        RayleighWaterOverlay=RayleighWater+MaskSkull*RayleighWater.max()/10
        
        
        self._InPeakValue[:,:,:self._ZSourceLocation+1]=0.0
        self._PhaseMap[:,:,:self._ZSourceLocation+1]=0.0
        self._PressMapFourier[:,:,:self._ZSourceLocation+1]=0.0
        
        
        FullSolutionPressure=np.zeros(self._SkullMaskDataOrig.shape,np.float32)
        FullSolutionPressure[self._XShrink_L:upperXR,
                      self._YShrink_L:upperYR,
                      self._ZShrink_L:upperZR]=\
                      self._InPeakValue[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset]
        FullSolutionPressure=np.flip(FullSolutionPressure,axis=2)

        FullSolutionPhase=np.zeros(self._SkullMaskDataOrig.shape,np.float32)
        FullSolutionPhase[self._XShrink_L:upperXR,
                      self._YShrink_L:upperYR,
                      self._ZShrink_L:upperZR]=\
                      self._PhaseMap[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset]
        FullSolutionPhase=np.flip(FullSolutionPhase,axis=2)


        MaskCalcRegions[self._XShrink_L:upperXR, self._YShrink_L:upperYR,self._ZShrink_L:upperZR ]=True
        MaskCalcRegions=np.flip(MaskCalcRegions,axis=2)
        FullSolutionPressureRefocus=np.zeros(self._SkullMaskDataOrig.shape,np.float32)
        FullSolutionPhaseRefocus=np.zeros(self._SkullMaskDataOrig.shape,np.float32)
        
        if bDoRefocusing:
            self._InPeakValueRefocus[:,:,:self._ZSourceLocation+1]=0.0
            self._PhaseMapRefocus[:,:,:self._ZSourceLocation+1]=0.0
            self._PressMapFourierRefocus[:,:,:self._ZSourceLocation+1]=0.0
            FullSolutionPressureRefocus[self._XShrink_L:upperXR,
                          self._YShrink_L:upperYR,
                          self._ZShrink_L:upperZR]=\
                          self._InPeakValueRefocus[self._XLOffset:-self._XROffset,
                                       self._YLOffset:-self._YROffset,
                                       self._ZLOffset:-self._ZROffset]
            FullSolutionPressureRefocus=np.flip(FullSolutionPressureRefocus,axis=2)
            FullSolutionPhaseRefocus[self._XShrink_L:upperXR,
                          self._YShrink_L:upperYR,
                          self._ZShrink_L:upperZR]=\
                          self._PhaseMapRefocus[self._XLOffset:-self._XROffset,
                                       self._YLOffset:-self._YROffset,
                                       self._ZLOffset:-self._ZROffset]
            FullSolutionPhaseRefocus=np.flip(FullSolutionPhaseRefocus,axis=2)
        
        DataForSim ={}
        DataForSim['p_amp']=self._InPeakValue[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset].copy()
        if self._bSaveStress or self._bSaveDisplacement:
            for k in self._DictPeakValue:
                if k != 'Pressure':
                    DataForSim[k] = self._DictPeakValue[k][self._XLOffset:-self._XROffset,
                                            self._YLOffset:-self._YROffset,
                                            self._ZLOffset:-self._ZROffset]
        DataForSim['p_complex']=self._PressMapFourier[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset].copy()
        
        if bDoRefocusing:
            DataForSim['p_amp_refocus']=self._InPeakValueRefocus[self._XLOffset:-self._XROffset,
                                       self._YLOffset:-self._YROffset,
                                       self._ZLOffset:-self._ZROffset].copy()
            DataForSim['p_complex_refocus']=self._PressMapFourierRefocus[self._XLOffset:-self._XROffset,
                                       self._YLOffset:-self._YROffset,
                                       self._ZLOffset:-self._ZROffset].copy()
        if self._DensityCTMap is not None:
            MaterialMap=self._MaterialMapNoCT.copy()
            DataForSim['MaterialMapCT']=self._MaterialMap[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset].copy()
        else:
            MaterialMap=self._MaterialMap.copy()
        OrigMaterialFocalSpot=MaterialMap[self._FocalSpotLocation[0],self._FocalSpotLocation[1],self._FocalSpotLocation[2]]
        MaterialMap[self._FocalSpotLocation[0],self._FocalSpotLocation[1],self._FocalSpotLocation[2]]=500.0
        
        DataForSim['MaterialMap']=MaterialMap[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset].copy()
        
        TargetLocation=np.array(np.where(DataForSim['MaterialMap']==500.0)).flatten()
        DataForSim['MaterialMap'][DataForSim['MaterialMap']==500.0]=OrigMaterialFocalSpot #we switch it back to soft tissue
        
        if bUseRayleighForWater:
            DataForSim['p_complex_water']=self._u2RayleighField[self._XLOffset:-self._XROffset,
                                   self._YLOffset:-self._YROffset,
                                   self._ZLOffset:-self._ZROffset]
            DataForSim['p_amp_water']=np.abs(DataForSim['p_complex_water'])
        for k in DataForSim:
            DataForSim[k]=np.flip(DataForSim[k],axis=2)
        DataForSim['Material']=self.ReturnArrayMaterial()
        DataForSim['x_vec']=self._XDim[self._XLOffset:-self._XROffset]
        DataForSim['y_vec']=self._YDim[self._YLOffset:-self._YROffset]
        DataForSim['z_vec']=self._ZDim[self._ZLOffset:-self._ZROffset]
        DataForSim['SpatialStep']=self._SpatialStep
        DataForSim['TargetLocation']=TargetLocation
        DataForSim['zLengthBeyonFocalPoint']=self._zLengthBeyonFocalPointWhenNarrow

        DataForSim['SourcePlane']=self._SourceMapRayleigh[self._PMLThickness:-self._PMLThickness,
                                                          self._PMLThickness:-self._PMLThickness]
        
        
        assert(np.all(np.array(RayleighWaterOverlay.shape)==np.array(FullSolutionPressure.shape)))
        return  RayleighWater,\
                RayleighWaterOverlay,\
                FullSolutionPressure,\
                FullSolutionPressureRefocus,\
                DataForSim,\
                MaskCalcRegions,\
                FullSolutionPhase,\
                FullSolutionPhaseRefocus,\
                RayleighWaterPhase
                