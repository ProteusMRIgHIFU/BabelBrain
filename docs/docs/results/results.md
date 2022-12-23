Results files
----
Multiple files are generated and will be saved in the same directory where the input T1W file is located. These are the most relevant files.

# 1 - Input mask in Nifti format

`<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_BabelViscoInput.nii.gz`

For example:

`STN_CTX_500_250kHz_6PPW_BabelViscoInput.nii.gz`

This file contains the 3D mask in T1W coordinates. The raw 3D array is arranged to be aligned to create a Cartesian domain aligned following the trajectory specified by the user. Skin, cortical, trabecular, cortical, and brain tissues are defined with values of 1, 2,3 and 4, respectively. 

<img src="Results -2.png" height=350px> 

If a ZTE or real CT scan was indicated as input, the files `<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_CT.nii.gz` and  `<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_CT.npz` will be also present. The `<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_CT.nii.gz` is a map of unique CT values (re-quantified in 12 bits). `<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_CT.npz` contains the indexing of unique CT map values to HU.

<img src="Results -3.png" height=350px> 

# 2 - Acoustic pressure field results in Nifti format

`<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_FullElasticSolution.nii.gz`

For example:

`STN_CTX_500_250kHz_6PPW_FullElasticSolution.nii.gz`

This file contains the 3D pressure field in T1W coordinates. A sub-sampled version (1:2 ratio) is also saved (`<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_FullElasticSolution_Sub.nii.gz`) for visualization tool that cannot handle very high-resolution Nifti files. Supplementary Nifti files showing results for benchmarking are also saved. For example results in water-only conditions, including with Rayleigh integral. Water-conditions results are also needed to calculate the required $I_{SPPA}$ value to use to program devices during real experimentations. These acoustic maps should be considered **adimensional** and are mostly used for visualization purposes. 

# 3 - STL files
BabelBrain will generate STL files of the transducer that will be in T1W coordinates and aligned using the planning and simulation information. 
 If using 'charm' from Simbnibs 4.0, BabelBrain will generate STL files for skin, csf and skull 
regions (`skin.stl`,`csf.stl`,`bone.stl`) in the directory specified for charm output.

Below there is an example of visualization of STL files of transducer, skin and acoustic maps overlay with planning T1W imaging.

<img src="Results -1.png" height=550px>

# 2 - Thermal and intensity maps:
`<Target ID>_<Tx ID>_<Frequency ID>_<PPW ID>_DataForSim-ThermalField-<Timming IDs>.h5` 

For example:
 `STN_CTX_500_500kHz_6PPW_DataForSim-ThermalField-Duration-40-DurationOff-40-DC-300-Isppa-5.0W-PRF-10Hz.h5`. 

These are HDF5 files (also saved in Matlab .mat format) that contains multiple variables and arrays. BabelBrain contains functions to read and write HDF5 files in a simplified way (similar to Matlab and npz format). Below there is simple code to printout the variables in the files.

```Python
from BabelViscoFDTD.H5pySimple import ReadFromH5py
import pprint
pp = pprint.PrettyPrinter(indent=4, width=50)
Data=ReadFromH5py('STN_CTX_500_500kHz_6PPW_DataForSim-ThermalField-Duration-40-DurationOff-40-DC-300-Isppa-5.0W-PRF-10Hz.h5')
pp.pprint(list(Data.keys()))
```
```
[   'AdjustmentInRAS',
    'CEMBrain',
    'CEMSkin',
    'CEMSkull',
    'DistanceFromSkin',
    'DoseEndFUS',
    'FinalDose',
    'FinalTemp',
    'Isppa',
    'Ispta',
    'MI',
    'MaterialList',
    'MaterialMap',
    'MaterialMap_central',
    'MaxBrainPressure',
    'MaxIsppa',
    'MaxIspta',
    'MonitorSlice',
    'RatioLosses',
    'TI',
    'TIC',
    'TIS',
    'TargetLocation',
    'TempEndFUS',
    'TempProfileTarget',
    'TemperaturePoints',
    'TimeProfileTarget',
    'TxMechanicalAdjustmentZ',
    'ZSteering',
    'dt',
    'mBrain',
    'mSkin',
    'mSkull',
    'p_map',
    'p_map_central',
    'x_vec',
    'y_vec',
    'z_vec']
```

These files are generated in function of the thermal profiles chosen for the simulations. There would be one file for each combination of duration, duty cycle and pulse repetition. Thermal maps and acoustic intensity are calculated assuming a $I_{SPPA}$ of 5 W/cm$^2$ at the target. BabelBrain scales results based on this intensity. Below there is a simple code showing how BabelBrain calculates and displays results scaled to a desired $I_{SPPA}$ of  10 W/cm$^2$.

```Python
import matplotlib.pyplot as plt
import numpy as np

def RCoeff(Temperature):
    R = np.ones(Temperature.shape)*0.25
    R[Temperature>=43]=0.5
    return R
#### Selected Isppa
SelIsppa=10.0
####


BaseIsppa = 5.0
DutyCycle=0.3

Loc=DataThermal['TargetLocation']
xf=DataThermal['x_vec']
zf=DataThermal['z_vec']

SelY=Loc[1]

SkinZ=np.array(np.where(DataThermal['MaterialMap'][:,SelY,:]==1)).T.min(axis=0)[1]
zf-=zf[SkinZ]

IsppaRatio=SelIsppa/BaseIsppa

PresRatio=np.sqrt(IsppaRatio)

AdjustedIsspa = SelIsppa/DataThermal['RatioLosses']

Ispta=SelIsppa*DutyCycle

AdjustedTemp=(DataThermal['TemperaturePoints']-37)*IsppaRatio+37

#thermal dose in minutes
DoseUpdate=np.sum(RCoeff(AdjustedTemp)**(43.0-AdjustedTemp),axis=1)*DataThermal['dt']/60
MI=DataThermal['MI']*PresRatio

MTB=DataThermal['TI']*IsppaRatio+37
MTC=DataThermal['TIC']*IsppaRatio+37
MTS=DataThermal['TIS']*IsppaRatio+37

CEMS=DoseUpdate[0]
CEMB=DoseUpdate[1]
CEMC=DoseUpdate[2]

print('Maximal temperature at skin, brain and skull = %3.2f, %3.2f, %3.2f' %(MTB,MTC,MTS))
print('Maximal thermal dose at skin, brain and skull', CEMS,CEMB,CEMC)


###
# We plot maps at target plane and ovel time 

DensityMap=DataThermal['MaterialList']['Density'][DataThermal['MaterialMap'][:,SelY,:]]
SoSMap=    DataThermal['MaterialList']['SoS'][DataThermal['MaterialMap'][:,SelY,:]]
IntensityMap=(DataThermal['p_map'][:,SelY,:]**2/2/DensityMap/SoSMap/1e4*IsppaRatio).T
IntensityMap[0,:]=0
Tmap=(DataThermal['TempEndFUS'][:,SelY,:]-37.0)*IsppaRatio+37.0
figure=plt.figure(figsize=(10, 4.5))
static_ax1,static_ax2 = figure.subplots(1,2)
IntensityIm=static_ax1.imshow(IntensityMap,extent=[xf.min(),xf.max(),zf.max(),zf.min()],
                cmap=plt.cm.jet)
static_ax1.plot(xf[Loc[0]],zf[Loc[2]],'k+',markersize=18)
static_ax1.set_title('Isppa (W/cm$^2$)')
plt.colorbar(IntensityIm,ax=static_ax1)

XX,ZZ=np.meshgrid(xf,zf)

contour1=static_ax1.contour(XX,ZZ,DataThermal['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)
static_ax1.set_ylabel('Distance from skin (mm)')
ThermalIm=static_ax2.imshow(Tmap.T,
                extent=[xf.min(),xf.max(),zf.max(),zf.min()],cmap=plt.cm.jet,vmin=37)
static_ax2.plot(xf[Loc[0]],zf[Loc[2]],'k+',markersize=18)
static_ax2.set_title('Temperature ($^{\circ}$C)')

plt.colorbar(ThermalIm,ax=static_ax2)
contour2=static_ax2.contour(XX,ZZ,DataThermal['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)

plt.figure(figsize=(10, 4.5))
plt.plot(DataThermal['TimeProfileTarget'],AdjustedTemp.T)
plt.legend(['Skin','Brain','Skull'])
plt.xlabel('time (s)');plt.ylabel('Temperature ($^{\circ}$C)')
plt.title('Maximal temperature')
```

```
Maximal temperature at skin, brain and skull = 37.67, 37.36, 37.27
Maximal thermal dose at skin, brain and skull 0.0006094899601355539 0.0007513694465752783 0.0006360966246583003
```

<img src="Results -4.png" height=650px>