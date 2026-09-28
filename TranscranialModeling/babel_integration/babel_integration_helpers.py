'''
Shared helper utilities for BabelBrain transcranial simulation pipelines.

Contains:
  - Tissue/bone acoustic material property functions (HU ↔ density ↔ SoS/attenuation)
  - Pre-computed per-frequency material tables (MatFreq)
  - Simulation utility functions (SDR calculation, NIfTI I/O, material map construction)
  - CT-derived acoustic property calculation (CalculateCTDerivedInfo, CreateMaterialMaps)
  - Output filename generation (OutputFileNames)

These are module-level utilities shared by all transducer-specific integration classes.
'''
import os
import platform
import subprocess
import warnings

import h5py
import nibabel
import numpy as np
import pandas as pd
import pwlf
import scipy
import SimpleITK as sitk
from scipy import interpolate

from BabelViscoFDTD.H5pySimple import ReadFromH5py

# Artifact recording (see BabelBrain/ArtifactIO.py). Guarded so this module still
# imports if ArtifactIO isn't on the path; a no-op unless BABEL_ARTIFACT_LOG is set.
# Duplicated here (rather than imported from babel_integration_base) to avoid a
# circular import: babel_integration_base imports * from this module.
try:
    from ArtifactIO import record as _rec_artifact
except ImportError:
    def _rec_artifact(_p, **_k):
        return _p
from Utils.paths import resource_path

np.seterr(divide='raise')
warnings.filterwarnings("ignore", category=DeprecationWarning)

_IS_MAC = platform.system() == 'Darwin'

## Global definitions

DbToNeper=1/(20*np.log10(np.exp(1)))

_MapPichardo = ReadFromH5py(os.path.join(resource_path(__file__).parent, 'MapPichardo.h5'))
if scipy.__version__>"1.14.0":
    interp2d=interpolate.RectBivariateSpline
    _PichardoSOS=interp2d(_MapPichardo['rho'], _MapPichardo['freq'], _MapPichardo['MapSoS'],kx=1,ky=1)
    _PichardoAtt=interp2d(_MapPichardo['rho'], _MapPichardo['freq'], _MapPichardo['MapAtt'],kx=1,ky=1)
else:
    interp2d=interpolate.interp2d
    _PichardoSOS=interp2d(_MapPichardo['rho'], _MapPichardo['freq'], _MapPichardo['MapSoS'])
    _PichardoAtt=interp2d(_MapPichardo['rho'], _MapPichardo['freq'], _MapPichardo['MapAtt'])

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
for f in np.arange(100e3,1125e3,5e3):
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
    

def DensityToHUBony(ct_in):
    '''
    Convert CT values to bone density using piecewise linear fitting.
    
    Parameters
    ----------
    ct_in : array-like or float
        CT values (Hounsfield units) for bone.
    
    Returns
    -------
    float or np.ndarray
        Predicted bone density (kg/m³).
    '''
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
    return pwf.predict(ct_in)
    

def HUtoDensityKWave(hu_in):
    '''
    Convert Hounsfield Units to density using k-Wave model.
    
    Adapted from hounsfield2density.m from k_wave
    References:
    - Phys. Med. Biol., 41, pp. 111-124 (1996).
    - Acoust. Res. Lett. Online, 1(2), pp. 37-42 (2000).
    
    Parameters
    ----------
    hu_in : array-like or float
        Hounsfield Unit values.
    
    Returns
    -------
    float or np.ndarray
        Density values (kg/m³).
    '''
    #Adapted from hounsfield2density.m fromk k_wave
    # Phys. Med. Biol., 41, pp. 111-124 (1996).
    # Acoust. Res. Lett. Online, 1(2), pp. 37-42 (2000). 
    HU = hu_in+1000
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

def HUtoDensityAirTissue(hu_in):
    '''
    Convert Hounsfield Units to density using linear air-tissue model.
    
    Linear fitting using reference points:
    - DensityAir = 1.293 kg/m³
    - DensityTissue = 1041 kg/m³
    - HUAir = -1000
    - HUTissue = 27
    
    Parameters
    ----------
    hu_in : array-like or float
        Hounsfield Unit values.
    
    Returns
    -------
    float or np.ndarray
        Density values (kg/m³).
    '''
    # linear fitting using
    # DensityAir=1.293 
    # DensityTissue=1041
    # HUAir=-1000
    # HUTissue=27
    
    pf=np.array([1.01237293, 1.01366593e+03])
    return np.polyval(pf,hu_in)

def HUtoDensityMarsac(hu_in):
    '''
    Convert Hounsfield Units to density using Marsac model.
    
    Linear normalization between air and bone density limits.
    
    Parameters
    ----------
    hu_in : array-like or float
        Hounsfield Unit values.
    
    Returns
    -------
    float or np.ndarray
        Density values (kg/m³).
    '''
    rhomin=1000.0
    rhomax=2700.0
    return rhomin+ (rhomax-rhomin)*hu_in/hu_in.max()

def HUtoDensityUCLLowEnergy(hu_in):
    '''
    Convert Hounsfield Units to density using UCL low-energy calibration model.
    
    Uses calibration data from https://github.com/ucl-bug/petra-to-ct
    
    Parameters
    ----------
    hu_in : array-like or float
        Hounsfield Unit values.
    
    Returns
    -------
    float or np.ndarray
        Density values (kg/m³) interpolated from calibration table.
    '''
    #using calibration reported in https://github.com/ucl-bug/petra-to-ct 
    f = h5py.File(os.path.join(resource_path(__file__).parent, 'ct-calibration-low-dose-30-March-2023-v1.h5'), 'r')
    ct_calibration=f['ct_calibration'][:][0,:,:].T
    return np.interp(hu_in,ct_calibration[0,:],ct_calibration[1,:])

def SimNIBS_PETRApct_Density(hu_in):
    MAX_CT_VALUE = 3150 # [hu]
    MAX_DENSITY_VALUE = 3147.35469785 # [kg/m3]
    DENSITY_WATER = 1000.0  # [kg/m3]
    points = np.loadtxt(os.path.join(resource_path(__file__).parent, 'ct_to_density_calibration_cph2025_line_v1.csv'), delimiter=",")
    points = np.concatenate((points, [[MAX_CT_VALUE, MAX_DENSITY_VALUE]]))
    hu_values, density_values = points[:, 0], points[:, 1]
    assert np.all(np.diff(hu_values) > 0), "ct to density values must be increasing only in the calibration file."
    assert np.all(np.diff(density_values) > 0), "ct to density values must be increasing only in the calibration file."

    hu_values, density_values = points[:, 0], points[:, 1]
    density = np.interp(hu_in, hu_values, density_values)
    density[(density < DENSITY_WATER)] = DENSITY_WATER # bone should not have lower density than water
    print('SimNIBS_PETRApct_Density, min, max',density.min(),density.max())
    return density


def DensitytoLSOSMarsac(density):
    '''
    Convert tissue density to longitudinal speed of sound using Marsac model.
    
    Linear mapping between minimum and maximum sound velocities based on density range.
    
    Parameters
    ----------
    density : array-like or float
        Tissue density (kg/m³).
    
    Returns
    -------
    float or np.ndarray
        Longitudinal speed of sound (m/s).
    '''
    cmin=1500.0
    cmax=3000.0
    return cmin+ (cmax-cmin)*(density-density.min())/(density.max()-density.min())

def DensityToLAttMcDannold(density, frequency):
    '''
    Convert density to longitudinal attenuation using McDannold model.
    
    Uses polynomial fitting and frequency scaling from McDannold reference.
    
    Parameters
    ----------
    density : array-like or float
        Tissue density (kg/m³).
    frequency : float
        Acoustic frequency (Hz). Attenuation is scaled relative to 660 kHz reference.
    
    Returns
    -------
    float or np.ndarray
        Longitudinal attenuation (Np/m).
    '''
    FreqReference =660e3
    poly=np.flip(np.array([5.71e3,-9.02, 5.40e-3,-1.41e-6,1.36e-10]))
    return np.polyval(poly,density)*frequency/FreqReference #we assume a linear relatinship

def DensityToLSOSMcDannold(density):
    '''
    Convert density to longitudinal speed of sound using McDannold model.
    
    Parameters
    ----------
    density : array-like or float
        Tissue density (kg/m³).
    
    Returns
    -------
    float or np.ndarray
        Longitudinal speed of sound (m/s).
    '''
    poly=np.flip(np.array([1.24e-3,-7.63e-7,1.69e-10,5.31e-16,-2.79e-18]))
    return 1.0/np.polyval(poly,density)

def HUtoPorosity(hu_in):
    '''
    Convert Hounsfield Units to bone porosity.
    
    Parameters
    ----------
    hu_in : array-like or float
        Hounsfield Unit values.
    
    Returns
    -------
    float or np.ndarray
        Porosity values (0-1).
    '''
    Phi = 1.0 - hu_in/hu_in.max()
    return Phi

def PorositytoDensity(phi):
    '''
    Convert bone porosity to density.
    
    Parameters
    ----------
    phi : array-like or float
        Porosity values (0-1).
    
    Returns
    -------
    float or np.ndarray
        Tissue density (kg/m³).
    '''
    Density = 1000.0 * phi + 2200*(1.0-phi)
    return Density

def PorositytoLSOS(phi):
    '''
    Convert bone porosity to longitudinal speed of sound.
    
    Parameters
    ----------
    phi : array-like or float
        Porosity values (0-1).
    
    Returns
    -------
    float or np.ndarray
        Longitudinal speed of sound (m/s).
    '''
    SoS = 1500 * phi + 3100*(1.0-phi)
    return SoS

def PorositytoLAtt(phi, frequency):
    '''
    Convert bone porosity to longitudinal attenuation.
    
    Parameters
    ----------
    phi : array-like or float
        Porosity values (0-1).
    frequency : float
        Acoustic frequency (Hz).
    
    Returns
    -------
    float or np.ndarray
        Longitudinal attenuation (Np/m).
    '''
    amin= 2.302555836 *frequency/1e6
    amax= 92.10223344 *frequency/1e6
    Att = amin + (amax - amin)*(phi**0.5)
    return Att

def HUtoAttenuationWebb(hu, frequency, params=['GE','120','B','','0.5, 0.6']):
    '''
    Convert Hounsfield Units to attenuation using Webb model.
    
    Parameters from Table IV in Webb et al. IEEE Trans Ultrason Ferroelectr Freq Control 68, no. 5 (2020): 1532-1545.
    DOI: 10.1109/TUFFC.2020.3039743
    
    Parameters
    ----------
    hu : array-like or float
        Hounsfield Unit values.
    frequency : float
        Acoustic frequency (Hz).
    params : list, optional
        Scanner parameters [Scanner, Energy, Kernel, Other, Resolution].
        Default is ['GE','120','B','','0.5, 0.6'] for GE 120 kVp BonePlus kernel.
    
    Returns
    -------
    float or np.ndarray
        Attenuation values (Np/m × 100).
    '''
    #these values are for 120 kVp, BonePlus Kernel, axial res = 0.49, slice res=0.63 in GE Scanners
    #Table IV in Webb et al. IEEE Trans Ultrason Ferroelectr Freq Control 68, no. 5 (2020): 1532-1545.
    # DOI: 10.1109/TUFFC.2020.3039743

    lst_str_cols = ['Scanner','Energy','Kernel','Other','Res']
    dict_dtypes = {x : 'str'  for x in lst_str_cols}

    df = pd.read_csv(os.path.join(resource_path(__file__).parent, 'WebbHU_Att.csv'), keep_default_na=False, index_col=lst_str_cols, dtype=dict_dtypes)
    
    sel=df.loc[[params]]

    print('Using Webb Att mapping with Params (Alpha_0, Beta, c)', 
          params, 
          sel.iloc[0]['Alpha_0']*100,
          sel.iloc[0]['Beta'],
          sel.iloc[0]['c'])

    return (sel.iloc[0]['Alpha_0']*(frequency/1e6)**sel.iloc[0]['Beta'] * np.exp(hu*(sel.iloc[0]['c'])))*100

def SpeedofSoundWebbDataset():
    '''
    Load and return Webb dataset for Hounsfield Unit to speed of sound mapping.
    
    Data from Tables I and II in Webb et al. IEEE Trans Ultrason Ferroelectr Freq Control. 2018 Jul; 65(7): 1111–1124. 
    DOI: 10.1109/TUFFC.2018.2827899
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by scanner parameters with speed of sound calibration data.
    '''
    lst_str_cols = ['Scanner','Energy','Kernel','Other','Res']
    dict_dtypes = {x : 'str'  for x in lst_str_cols}

    df = pd.read_csv(os.path.join(resource_path(__file__).parent, 'WebbHU_SoS.csv'), keep_default_na=False, index_col=lst_str_cols, dtype=dict_dtypes)
    return df


def HUtoLongSpeedofSoundWebb(hu, params=['GE','120','B','','0.5, 0.6']):
    '''
    Convert Hounsfield Units to longitudinal speed of sound using Webb model.
    
    Parameters from Tables I and II in Webb et al. IEEE Trans Ultrason Ferroelectr Freq Control. 2018 Jul; 65(7): 1111–1124. 
    DOI: 10.1109/TUFFC.2018.2827899
    
    Parameters
    ----------
    hu : array-like or float
        Hounsfield Unit values.
    params : list, optional
        Scanner parameters [Scanner, Energy, Kernel, Other, Resolution].
        Default is ['GE','120','B','','0.5, 0.6'] for GE 120 kVp BonePlus kernel.
    
    Returns
    -------
    float or np.ndarray
        Longitudinal speed of sound (m/s).
    '''
    #these values are for 120 kVp, BonePlus Kernel, axial res = 0.49, slice res=0.63 in GE Scanners
    #Tables I and II in Webb et al. IEEE Trans Ultrason Ferroelectr Freq Control. 2018 Jul; 65(7): 1111–1124. 
    # DOI: 10.1109/TUFFC.2018.2827899

    df=SpeedofSoundWebbDataset()
    
    sel=df.loc[[params]]

    print('Using Webb  SOS mapping with Params (slope, intercept)', 
          params, 
          sel.iloc[0]['Slope'],
          sel.iloc[0]['Intercept']*1000.0)

    return sel.iloc[0]['Slope']*hu + sel.iloc[0]['Intercept']*1000.0


def DensityToLSOSPichardo(density, frequency):
    '''
    Convert density to longitudinal speed of sound using Pichardo model.
    
    Parameters
    ----------
    density : array-like or float
        Tissue density (kg/m³).
    frequency : float
        Acoustic frequency (Hz).
    
    Returns
    -------
    float or np.ndarray
        Longitudinal speed of sound (m/s).
    '''
    return _PichardoSOS(density, frequency/1e6)

def DensityToLAttPichardo(density, frequency):
    '''
    Convert density to longitudinal attenuation using Pichardo model.
    
    Parameters
    ----------
    density : array-like or float
        Tissue density (kg/m³).
    frequency : float
        Acoustic frequency (Hz).
    
    Returns
    -------
    float or np.ndarray
        Longitudinal attenuation (Np/m).
    '''
    return _PichardoAtt(density, frequency/1e6)

def DensityToSSoSPichardo(density):
    '''
    Convert density to shear speed of sound using Pichardo model.
    
    Based on Physics in Medicine & Biology, vol. 62, no. 17, p 6938, 2017.
    Uses average of values at two reported frequencies.
    
    Parameters
    ----------
    density : array-like or float
        Tissue density (kg/m³).
    
    Returns
    -------
    float or np.ndarray
        Shear speed of sound (m/s).
    '''
    #using Physics in Medicine & Biology, vol. 62, bo. 17,p 6938, 2017, we average the values for the two reported frequencies
    return density*0.422 + 680.515  
    
def make_affine_itk_friendly(affine, eps=1e-12, report=True):
    '''
    Create an orthonormal affine matrix with preserved voxel sizes.
    
    Return a copy of affine whose top-left 3x3 is orthonormal up to column scaling 
    (i.e. preserves voxel sizes).
    
    Parameters
    ----------
    affine : (4, 4) np.ndarray
        Affine transformation matrix.
    eps : float, optional
        Tiny tolerance for near-zero scales (default is 1e-12).
    report : bool, optional
        If True, returns diagnostics along with affine (default is True).
    
    Returns
    -------
    np.ndarray or tuple
        If report is False: (4, 4) orthonormalized affine matrix.
        If report is True: tuple of (orthonormalized affine, diagnostics dict).
    '''
    """
    Return a copy of `affine` whose top-left 3x3 is orthonormal
    up to column scaling (i.e. preserves voxel sizes).
    - affine: (4,4) array
    - eps: tiny tolerance for near-zero scales
    - report: if True, returns diagnostics along with affine
    """
    A = affine.copy()
    M = A[:3, :3].astype(float)

    # Column norms = voxel scales (for NIfTI affine convention)
    scales = np.linalg.norm(M, axis=0)  # length for each axis vector (3,)
    # Protect against zero/near-zero scale
    tiny = scales < eps
    if np.any(tiny):
        raise ValueError(f"Detected near-zero scale(s) on axes: {np.where(tiny)[0].tolist()}")

    # Direction matrix (each column is the direction cosine)
    D = M / scales[np.newaxis, :]  # shape (3,3)

    # Find nearest orthonormal matrix to D using SVD
    U, svals, Vt = np.linalg.svd(D)
    R = U @ Vt

    # Ensure right-handed coordinate system unless original had reflection
    det_original = np.linalg.det(D)
    det_R = np.linalg.det(R)
    if det_R < 0:
        # Fix reflection: flip sign of last column of U then recompute
        U[:, -1] *= -1
        R = U @ Vt
        det_R = np.linalg.det(R)

    # Reconstruct matrix preserving scales (columns)
    M_new = R * scales[np.newaxis, :]

    # Place back
    A_clean = A.copy()
    A_clean[:3, :3] = M_new

    if not report:
        return A_clean

    # Diagnostics
    # 1) How non-orthonormal original was: check D^T D vs I
    orig_gram = D.T @ D
    orig_offdiag = orig_gram - np.eye(3)
    orig_max_off = np.max(np.abs(orig_offdiag))

    # 2) After cleaning: check orthonormality of R (should be ~I)
    R_gram = R.T @ R
    R_offdiag = R_gram - np.eye(3)
    R_max_off = np.max(np.abs(R_offdiag))

    # 3) Change in matrix
    max_abs_change = np.max(np.abs(M_new - M))
    max_rel_change = np.max(np.abs((M_new - M) / (np.where(np.abs(M) > 0, M, 1.0))))

    return {
        "affine": A_clean,
        "scales_before": scales,
        "det_direction_before": det_original,
        "det_direction_after": det_R,
        "orig_max_offdiag": float(orig_max_off),
        "R_max_offdiag": float(R_max_off),
        "max_abs_change_in_3x3": float(max_abs_change),
        "max_rel_change_in_3x3": float(max_rel_change),
    }

def SaveNiftiEnforcedISO(nii_in, fn):
    '''
    Save NIfTI image with isotropic voxel spacing using SimpleITK.
    
    Attempts to enforce isotropic spacing; if affine is not orthonormal,
    attempts to fix it or falls back to flirt registration.
    
    Parameters
    ----------
    nii_in : nibabel.Nifti1Image
        Input NIfTI image to save.
    fn : str
        Output filename path.
    '''
    fn_unc=fn.split('.gz')[0] #we save uncompressed for faster operation
    nii_in.to_filename(fn_unc)
    newfn=fn.split('__.nii.gz')[0]+'.nii.gz'
    res = float(np.round(np.array(nii_in.header.get_zooms()).mean(),5))
    try:
        pre=sitk.ReadImage(fn_unc)
        pre.SetSpacing([res,res,res])
        sitk.WriteImage(pre, newfn)
        os.remove(fn_unc)
    except:
        try: #lets try with a clean affine
            print('Affine matrix was not exactly orthonormal, fixing affine matrix')
            aff_clean = make_affine_itk_friendly(nii_in.affine, report=False)
            nii = nibabel.Nifti1Image(nii_in.get_fdata(dtype=np.float32), aff_clean, header=nii_in.header)
            nii.to_filename(fn_unc)
            pre=sitk.ReadImage(fn_unc)
            pre.SetSpacing([res,res,res])
            sitk.WriteImage(pre, newfn)
            os.remove(fn_unc)
        except: #last resource is to use flirt
            res = '%6.5f' % (res)
            rcmd=['flirt','-in',fn_unc,'-ref',fn_unc,'-applyisoxfm',res,'-nosearch','-out','newfn']
            result = subprocess.run(rcmd, capture_output=True, text=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            assert(result.returncode==0)
            os.remove(fn_unc)

    _rec_artifact(newfn)


def ResaveNormalized(rpath, mask,bApplyOnlyMask=False):
    '''
    Resave simulation results with normalization based on mask.
    
    Normalizes result values to [0, 1] range and masks out regions outside mask region 4+.
    Saves to file with _NORM suffix.
    
    Parameters
    ----------
    rpath : str
        Path to input result file (must contain '_Sub.nii.gz').
    mask : nibabel.Nifti1Image
        Mask image for normalization region selection.
    '''
    assert('_Sub.nii.gz' in rpath)
    NRPath=rpath.replace('_Sub.nii.gz','_Sub_NORM.nii.gz')

    Results=nibabel.load(rpath)

    ResultsData=Results.get_fdata(dtype=np.float32)
    MaskData=mask.get_fdata(dtype=np.float32)
    ii,jj,kk=np.mgrid[0:ResultsData.shape[0],0:ResultsData.shape[1],0:ResultsData.shape[2]]

    Indexes=np.c_[(ii.flatten().T,jj.flatten().T,kk.flatten().T,np.ones((kk.size,1)))].T

    PosResults=Results.affine.dot(Indexes)

    IndexesMask=np.round(np.linalg.inv(mask.affine).dot(PosResults)).astype(int)
    IndexesMask[0,IndexesMask[0,:]>=MaskData.shape[0]]=MaskData.shape[0]-1
    IndexesMask[1,IndexesMask[1,:]>=MaskData.shape[1]]=MaskData.shape[1]-1
    IndexesMask[2,IndexesMask[2,:]>=MaskData.shape[2]]=MaskData.shape[2]-1

    SubMask=MaskData[IndexesMask[0,:],IndexesMask[1,:],IndexesMask[2,:]].reshape(ResultsData.shape)
    ResultsData[SubMask<4]=0
    if not bApplyOnlyMask:
        ResultsData/=ResultsData.max()
    NormalizedNifti=nibabel.Nifti1Image(ResultsData.astype(np.float32),Results.affine,header=Results.header)
    NormalizedNifti.to_filename(NRPath)
    _rec_artifact(NRPath)

def compute_sdr_from_rays(volume, skull_mask, spacing_mm=(1.0, 1.0),
                                      ray_spacing_mm=1.8, min_skull_voxels=3,center_region=0.5):
    """
    Casts rays along `axis` through skull voxels, spaced ~1.8 mm apart.
    Returns mean SDR across all rays.
    
    Per literature (SkullGAN paper, arxiv 2308.00206):
        SDR_per_ray = min(HU along ray) / max(HU along ray)
        SDR_global  = mean over all rays
    """
    rows, cols,depth = volume.shape
    step_i = max(1, int(round(ray_spacing_mm / spacing_mm[0])))
    step_j = max(1, int(round(ray_spacing_mm / spacing_mm[1])))
    
    sdr_values = []
    for r in range(0, rows, step_i):
        for c in range(0, cols, step_j):
            ray_hu = volume[r, c,:]
            ray_skull = skull_mask[r, c,:]
            
            skull_indices = np.where(ray_skull)[0]
            if skull_indices.size < min_skull_voxels:
                continue
            
            # Midpoint index of the contiguous skull segment
            mid_idx = len(skull_indices) // 2
            l_half = float(len(skull_indices))*center_region
            beg_indx=np.max([0,int(np.round(mid_idx-l_half/2))])
            end_idx=np.min([len(skull_indices)-1,1+int(np.round(mid_idx+l_half/2))])

            hu_center = ray_hu[skull_indices[beg_indx]:skull_indices[end_idx]].min()
            
            skull_hu = ray_hu[skull_indices]
            hu_max = skull_hu.max()

            if hu_max > 0:
                sdr_values.append(hu_center / hu_max)
    print("len of SDR",len(sdr_values))
    return np.mean(sdr_values) if sdr_values else float('nan')
    
####
bGPU_INITIALIZED = False
###

def CalculateCTDerivedInfo(CTFNAME,
                           Frequency,
                           bBrainSegmentation,
                           MappingMethod='Webb-Marsac',
                           CTMapCombo=('GE','120','B','','0.5, 0.6'),
                           bDensity=False,
                           bPETRA=False,
                           AIRMASK=None):
    '''
    Compute the CT/pseudo-CT derived acoustic information for the bone region.

    This is a standalone extraction of the block inside
    `RUN_SIM_BASE.GenerateSimulation` that, when a CT file is provided, loads
    the indexed CT map and derives per-index density, longitudinal speed of
    sound and longitudinal attenuation according to the selected mapping
    method. It is provided as an independent function so the logic can be
    reused/inspected; the caller is still responsible for the guard that
    decides whether CT-derived info is needed at all
    (`self._CTFNAME is not None and not bWaterOnly and ...`).

    Inputs
    ------
    CTFNAME : str or (array, array)
        Either a path to the indexed CT NIfTI file (maps to `self._CTFNAME`),
        in which case the indexed map is loaded and flipped along axis 2 and the
        unique HU values are read from the companion calibration file
        (`<prefix>CT-cal.npz`, key `UniqueHU`); or a two-element list/tuple
        `(IndexedCTMap, UniqueHU)` where `IndexedCTMap` is a 3D array assigned
        directly to `DensityCTMap` (no loading/flipping) and `UniqueHU` is
        assigned directly to `AllBoneHU`. The array form lets callers supply the
        indexed map and HU table in memory.
    Frequency : float
        Acoustic frequency in Hz. Maps to `self._Frequency`.
    bBrainSegmentation : bool
        Whether the skull mask contains segmented brain tissues (white/gray/CSF).
        Controls the material-index offset added to `DensityCTMap`.
    MappingMethod : str
        HU/density -> acoustic property mapping method. Maps to
        `self._MappingMethod`. One of 'Webb-Marsac', 'Aubry', 'Pichardo',
        'McDannold', 'Marsac-Aubry', 'Pichardo-Marsac', 'McDannold-Marsac'.
    CTMapCombo : list
        Webb dataset parameters used by the 'Webb-Marsac' method. Maps to
        `self._CTMapCombo`.
    bDensity : bool
        If True, the calibration file already holds densities (converted to HU
        internally). Maps to `self._bDensity`.
    bPETRA : bool
        If True (and not bDensity), use the SimNIBS PETRA-to-density formula for
        the 'Webb-Marsac' method. Maps to `self._bPETRA`.
    AIRMASK : str,np.ndarray or None
        Optional path to an air-region NIfTI file or pre-loaded 3D array. Maps to `self._AIRMASK`.

    Returns
    -------
    DensityCTMap : np.ndarray (uint32)
        Indexed CT map (flipped along axis 2) with the material-index offset
        applied (+6 with brain segmentation, +3 otherwise). Maps to the local
        `DensityCTMap` consumed by `CreateMaterialMaps`/`UpdateConditions`.
    DensitCTMapOrig : np.ndarray (uint32)
        Copy of the indexed CT map before the offset is applied (used later to
        recover HU values for the SDR calculation). Maps to `DensitCTMapOrig`.
    AirRegions : np.ndarray or None
        Indexed air-region map (flipped along axis 2) if `AIRMASK` is provided,
        else None. Maps to `AirRegions`.
    AllBoneHU : np.ndarray
        Unique HU values for the bone indices (after density->HU conversion when
        `bDensity` is True). Maps to `AllBoneHU`.
    DensityCTIT : np.ndarray
        Per-index density values for the bone region. Maps to `DensityCTIT`.
    LSoSIT : np.ndarray
        Per-index longitudinal speed of sound for the bone region. Maps to
        `LSoSIT`.
    LAttIT : np.ndarray
        Per-index longitudinal attenuation for the bone region. Maps to `LAttIT`.
    '''
    AirRegions=None
    if isinstance(CTFNAME,(list,tuple)):
        #in-memory inputs: (indexed CT map, unique HU table) provided directly
        DensityCTMap = np.flip(CTFNAME[0].astype(np.uint32),axis=2)
        DensitCTMapOrig=DensityCTMap.copy()
        AllBoneHU = CTFNAME[1]
    else:
        DensityCTMap = np.flip(nibabel.load(CTFNAME).get_fdata().astype(np.uint32),axis=2)
        DensitCTMapOrig=DensityCTMap.copy()
        AllBoneHU = np.load(CTFNAME.split('CT.nii.gz')[0]+'CT-cal.npz')['UniqueHU']
    if AIRMASK is not None:
        if type(AIRMASK) is str:
            AirRegions = np.flip(nibabel.load(AIRMASK).get_fdata().astype(np.uint32),axis=2)
        else:
            AirRegions = np.flip(AIRMASK.astype(np.uint32),axis=2)
    print('Range HU CT, Unique entries',AllBoneHU.min(),AllBoneHU.max(),len(AllBoneHU))
    print('USING MAPPING METHOD = ',MappingMethod)

    if bDensity:
        print('Density map specified, converting Density to HU')
        DensityCTIT= AllBoneHU.copy()
        print('min, max Density',DensityCTIT.min(),DensityCTIT.max())
        AllBoneHU = DensityToHUBony(DensityCTIT)
        print('min, max HU',AllBoneHU.min(),AllBoneHU.max())


    Porosity=HUtoPorosity(AllBoneHU)
    if MappingMethod=='Webb-Marsac':
        if bDensity == False:
            if bPETRA: #we use Bjorn's formula to convert to Density
                print('Using SimNIBS petra to density')
                DensityCTIT=SimNIBS_PETRApct_Density(AllBoneHU)
            else:
                print('Using 120 Kvp CT settings')
                DensityCTIT=HUtoDensityMarsac(AllBoneHU)
        print('Using CT combination', CTMapCombo)
        LSoSIT = HUtoLongSpeedofSoundWebb(AllBoneHU,params=CTMapCombo)
        LAttIT = HUtoAttenuationWebb(AllBoneHU,Frequency,params=CTMapCombo)
    elif MappingMethod=='Aubry':
        if bDensity == False:
            DensityCTIT = PorositytoDensity(Porosity)
        LSoSIT = PorositytoLSOS(Porosity)
        LAttIT = PorositytoLAtt(Porosity,Frequency)
    elif  MappingMethod=='Pichardo':
        if bDensity == False:
            DensityCTIT=HUtoDensityAirTissue(AllBoneHU)
        LSoSIT=DensityToLSOSPichardo(DensityCTIT,Frequency)
        LAttIT=DensityToLAttPichardo(DensityCTIT,Frequency)
    elif MappingMethod=='McDannold':
        if bDensity == False:
            DensityCTIT=HUtoDensityAirTissue(AllBoneHU)
        LSoSIT=DensityToLSOSMcDannold(DensityCTIT)
        LAttIT=DensityToLAttMcDannold(DensityCTIT,Frequency)
    #these are more experimental
    elif MappingMethod=='Marsac-Aubry':
        #Marsac did not calculate attenuation... we use Aubry's old
        if bDensity == False:
            DensityCTIT=HUtoDensityMarsac(AllBoneHU)
        LSoSIT=DensitytoLSOSMarsac(DensityCTIT)
        LAttIT = PorositytoLAtt(AllBoneHU,Frequency)
    elif MappingMethod=='Pichardo-Marsac':
        #Marsac did not calculate attenuation... we use Aubry's old
        if bDensity == False:
            DensityCTIT=HUtoDensityMarsac(AllBoneHU)
        LSoSIT=DensityToLSOSPichardo(DensityCTIT,Frequency)
        LAttIT=DensityToLAttPichardo(DensityCTIT,Frequency)
    elif MappingMethod=='McDannold-Marsac':
        #Marsac did not calculate attenuation... we use Aubry's old
        if bDensity == False:
            DensityCTIT=HUtoDensityMarsac(AllBoneHU)
        LSoSIT=DensityToLSOSMcDannold(DensityCTIT)
        LAttIT=DensityToLAttMcDannold(DensityCTIT,Frequency)
    else:
        raise ValueError('Unknown mapping method -' +MappingMethod )

    if bBrainSegmentation:
        DensityCTMap+=6  #The material index needs to add 3 to account water, skin, brain (non specific), white matter, gray matter and CSF
    else:
        DensityCTMap+=3 # The material index needs to add 3 to account water, skin and brain
    print("maximum CT index map value",DensityCTMap.max())
    print(" CT Map unique values",np.unique(DensityCTMap).shape)

    return DensityCTMap,DensitCTMapOrig,AirRegions,AllBoneHU,DensityCTIT,LSoSIT,LAttIT

def CreateMaterialMaps(N1,N2,N3,
                       SkullMaskDataOrig,
                       XLOffset,XROffset,YLOffset,YROffset,ZLOffset,ZROffset,
                       XShrink_L,upperXR,YShrink_L,upperYR,ZShrink_L,upperZR,
                       ZSourceLocation,
                       ArrayMaterial,
                       bWaterOnly=False,
                       bForceHomogenousMedium=False,
                       BenchmarkTestFile='',
                       DensityCTMap=None,
                       AirRegions=None):
    '''
    Build the simulation material maps from the (cropped) skull mask.

    This is a standalone extraction of the material-map preparation block that
    lives inside `SimulationConditionsBASE.UpdateConditions` (the code between
    the creation of `self._MaterialMapRef` and the `else` branch that fills
    `self._MaterialMap` with zeros). It is provided here so the logic can be
    inspected/validated before replacing the in-place code in UpdateConditions.

    Inputs
    ------
    N1,N2,N3 : int
        Shape of the full (padded) simulation domain. Maps to
        `self._N1,self._N2,self._N3`.
    SkullMaskDataOrig : np.ndarray
        Original (uncropped) skull/tissue label volume. Maps to
        `self._SkullMaskDataOrig`.
    XLOffset,XROffset,YLOffset,YROffset,ZLOffset,ZROffset : int
        Padding offsets into the simulation domain. Map to the
        `self._X/Y/Z L/R Offset` attributes.
    XShrink_L,upperXR,YShrink_L,upperYR,ZShrink_L,upperZR : int
        Cropping bounds into `SkullMaskDataOrig` (the `upper*R` values are the
        precomputed upper bounds, i.e. `self._upperXR` etc.). Map to the
        `self._*Shrink_L` and `self._upper*R` attributes.
    ZSourceLocation : int
        Z index of the source plane; tissue layers up to and including this
        index are replaced by water. Maps to `self._ZSourceLocation`.
    ArrayMaterial : np.ndarray
        Material property array used only to bound-check the CT density labels;
        equivalent to `self.ReturnArrayMaterial()`. Only `.shape[0]` is used.
    bWaterOnly : bool
        If True, produce a water-only (all zeros) material map.
    bForceHomogenousMedium : bool
        Testing flag; when True the material map is left at zeros here (the
        homogeneous fill happens downstream).
    BenchmarkTestFile : str
        Testing flag; a non-empty path means the material map is provided
        externally, so only the all-zeros map is produced here.
    DensityCTMap : np.ndarray or None
        Pseudo-CT density labels (uint32) to inject into the bone region. Maps
        to `self._DensityCTMap`.
    AirRegions : np.ndarray or None
        Air region labels to crop into the domain. Maps to `self._AirRegions`.

    Returns
    -------
    MaterialMap : np.ndarray (uint32)
        The working material map. Maps to `self._MaterialMap`.
    MaterialMapRef : np.ndarray (uint32)
        The reference (raw, label-preserving) material map. Maps to
        `self._MaterialMapRef`.
    MaterialMapNoCT : np.ndarray or None
        Copy of the material map before CT density injection (only produced
        when `DensityCTMap` is not None, else None). Maps to
        `self._MaterialMapNoCT`.
    SubAirRegions : np.ndarray or None
        Cropped air-region map (only produced when both `DensityCTMap` and
        `AirRegions` are not None, else None). Maps to `self._SubAirRegions`.
    '''
    MaterialMapNoCT=None
    SubAirRegions=None

    # Use explicit upper bounds instead of negative indexing (e.g. XLOffset:-XROffset)
    # so the slices stay correct when any *ROffset is 0. With negative indexing,
    # -XROffset would become -0==0 and collapse the slice to an empty range.
    upperXOff=N1-XROffset
    upperYOff=N2-YROffset
    upperZOff=N3-ZROffset

    MaterialMapRef=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size
    MaterialMapRef[XLOffset:upperXOff,
                   YLOffset:upperYOff,
                   ZLOffset:upperZOff]=\
                   SkullMaskDataOrig.astype(np.uint32)[XShrink_L:upperXR,
                                                       YShrink_L:upperYR,
                                                       ZShrink_L:upperZR]
    if bWaterOnly==False and bForceHomogenousMedium == False and len(BenchmarkTestFile)==0:
        MaterialMap=MaterialMapRef
        bBrainSegmentation = np.any(MaterialMap>5)
        if DensityCTMap is not None:
            assert(DensityCTMap.dtype==np.uint32)
            BoneRegion=(MaterialMap==2) | (MaterialMap==3)
            MaterialMapNoCT=MaterialMap.copy()
            if bBrainSegmentation:
                #we re arrange labels
                MaterialMap[MaterialMap==4]=2
                MaterialMap[MaterialMap==5]=2 #we define target as regular brain (we wil need to fix this later)
                MaterialMap[MaterialMap>=6]-=3
            else:
                MaterialMap[MaterialMap>=4]=2 # Brain region is in material 2
            SubCTMap=np.zeros_like(MaterialMap)
            SubCTMap[XLOffset:upperXOff,
                     YLOffset:upperYOff,
                     ZLOffset:upperZOff]=\
                       DensityCTMap[XShrink_L:upperXR,
                                    YShrink_L:upperYR,
                                    ZShrink_L:upperZR]
            MaterialMap[BoneRegion]=SubCTMap[BoneRegion]
            if AirRegions is not None:
                SubAirRegions=np.zeros_like(MaterialMap)
                SubAirRegions[XLOffset:upperXOff,
                              YLOffset:upperYOff,
                              ZLOffset:upperZOff]=\
                                  AirRegions[XShrink_L:upperXR,
                                             YShrink_L:upperYR,
                                             ZShrink_L:upperZR]
            assert(SubCTMap[BoneRegion].min()>=3)
            assert(SubCTMap[BoneRegion].max()<=ArrayMaterial.shape[0])

        else:
            if bBrainSegmentation:
                MaterialMap[MaterialMap>=5]-=1
            else:
                MaterialMap[MaterialMap==5]=4 # this is to make the focal spot location as brain tissue

        #We remove tissue layers
        MaterialMap[:,:,:ZSourceLocation+1] = 0 # we remove tissue layers by putting water
    else:
        MaterialMap=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size

    return MaterialMap,MaterialMapRef,MaterialMapNoCT,SubAirRegions


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
    OUT_FNAMES['FullElasticSolutionRefocusPhase_Sub']=CPREFIX+'FullElasticSolutionRefocusPhase_Sub.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocus__']=CPREFIX+'FullElasticSolutionRefocus__.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocusPhase__']=CPREFIX+'FullElasticSolutionRefocusPhase__.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocus_Sub__']=CPREFIX+'FullElasticSolutionRefocus_Sub__.nii.gz'
    OUT_FNAMES['FullElasticSolutionRefocusPhase_Sub__']=CPREFIX+'FullElasticSolutionRefocusPhase_Sub__.nii.gz'
    OUT_FNAMES['FullElasticSolution']=CPREFIX+'FullElasticSolution.nii.gz'
    OUT_FNAMES['FullElasticSolution_Sub']=CPREFIX+'FullElasticSolution_Sub.nii.gz'
    OUT_FNAMES['FullElasticSolutionPhase_Sub']=CPREFIX+'FullElasticSolutionPhase_Sub.nii.gz'
    OUT_FNAMES['FullElasticSolution__']=CPREFIX+'FullElasticSolution__.nii.gz'
    OUT_FNAMES['FullElasticSolutionPhase__']=CPREFIX+'FullElasticSolutionPhase__.nii.gz'
    OUT_FNAMES['FullElasticSolution_Sub__']=CPREFIX+'FullElasticSolution_Sub__.nii.gz'
    OUT_FNAMES['FullElasticSolutionPhase_Sub__']=CPREFIX+'FullElasticSolutionPhase_Sub__.nii.gz'
    OUT_FNAMES['DataForSim']=CPREFIX+'DataForSim.h5'
    return OUT_FNAMES

