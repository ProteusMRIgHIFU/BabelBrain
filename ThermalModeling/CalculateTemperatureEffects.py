'''
Tools to calculate thermal simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 23, 2021
     last update   - May 22, 2022

'''

import numpy as np
import matplotlib.pyplot as plt
from  scipy.io import loadmat,savemat
from BabelViscoFDTD.tools.RayleighAndBHTE import BHTE,BHTEMultiplePressureFields
from BabelViscoFDTD.H5pySimple import SaveToH5py,ReadFromH5py
from scipy.io import loadmat,savemat
from platform import platform
from os.path import isfile

def GetThermalOutName(InputPData,DurationUS,DurationOff,DutyCycle,Isppa,PRF):
    return InputPData.split('.h5')[0]+'-ThermalField-Duration-%i-DurationOff-%i-DC-%i-Isppa-%2.1fW-PRF-%iHz' % (DurationUS,DurationOff,DutyCycle*1000,Isppa,PRF)


def CalculateTemperatureEffects(InputPData,
                                DutyCycle=0.3,
                                Isppa=5,
                                sel_p='p_amp',
                                bPlot=True,
                                PRF=1500,
                                DurationUS=40,
                                DurationOff=40,
                                bForceRecalc=False,
                                OutTemperature=37,
                                bCalculateLosses=False,
                                Backend='CUDA'):#this will help to calculate the final voltage to apply during experiments

    
    outfname=GetThermalOutName(InputPData,DurationUS,DurationOff,DutyCycle,Isppa,PRF)

    print(outfname)

    print('Thermal sim with Backend',Backend)
    if bForceRecalc==False:
        if isfile(outfname+'.h5'):
            print('skipping', outfname)
            return outfname
    dt=0.01
    Input=ReadFromH5py(InputPData)
    MaterialList={}
    MaterialList['Density']=Input['Material'][:,0]
    MaterialList['SoS']=Input['Material'][:,1]
    MaterialList['Attenuation']=Input['Material'][:,3]
    if 'MaterialMapCT' not in Input:
        #Water, Skin, Cortical, Trabecular, Brain

        #https://itis.swiss/virtual-population/tissue-properties/database/heat-capacity/
        MaterialList['SpecificHeat']=[4178,3391,1313,2274,3630] #(J/kg/°C)
        #https://itis.swiss/virtual-population/tissue-properties/database/thermal-conductivity/
        MaterialList['Conductivity']=[0.6,0.37,0.32,0.31,0.51] # (W/m/°C)
        #https://itis.swiss/virtual-population/tissue-properties/database/heat-transfer-rate/
        MaterialList['Perfusion']=np.array([0,106,10,30,559])
        
        MaterialList['Absorption']=np.array([0,0.85,0.16,0.15,0.85])

        MaterialList['InitTemperature']=[OutTemperature,37,37,37,37]
    else:
        #Water, Skin, Brain and skull material
        MaterialList['SpecificHeat']=np.zeros_like(MaterialList['SoS'])
        MaterialList['SpecificHeat'][0:3]=[4178,3391,3630]
        MaterialList['SpecificHeat'][3:]=(1313+2274)/2

        MaterialList['Conductivity']=np.zeros_like(MaterialList['SoS'])
        MaterialList['Conductivity'][0:3]=[0.6,0.37,0.51]
        MaterialList['Conductivity'][3:]=(0.32+0.31)/2

        MaterialList['Perfusion']=np.zeros_like(MaterialList['SoS'])
        MaterialList['Perfusion'][0:3]=[0,106,559]
        MaterialList['Perfusion'][3:]=(10+30)/2

        MaterialList['Absorption']=np.zeros_like(MaterialList['SoS'])
        MaterialList['Absorption'][0:3]=[0,0.85,0.85]
        MaterialList['Absorption'][3:]=(0.16+0.15)/2

        MaterialList['InitTemperature']=np.zeros_like(MaterialList['SoS'])
        MaterialList['InitTemperature'][0]=OutTemperature
        MaterialList['InitTemperature'][1:]=37
    

    SaveDict={}
    SaveDict['MaterialList']=MaterialList

    
    nFactorMonitoring=int(50e-3/dt) # we just track every 50 ms
    TotalDurationSteps=int((DurationUS+.001)/dt)
    nStepsOn=int(DurationUS/dt) 
    TotalDurationStepsOff=int((DurationOff+.001)/dt)

    xf=Input['x_vec']
    yf=Input['y_vec']
    zf=Input['z_vec']
    
    pAmp=np.ascontiguousarray(np.flip(Input[sel_p],axis=2))
    print('pAmp.shape',pAmp.shape)
    
    if 'MaterialMapCT' in Input:
        MaterialMap=np.ascontiguousarray(np.flip(Input['MaterialMapCT'],axis=2))
    else:
        MaterialMap=np.ascontiguousarray(np.flip(Input['MaterialMap'],axis=2))
    
    LocIJK=Input['TargetLocation'].flatten()
    if 'MaterialMapCT' in Input:
        BrainID=2
        LimSoft=3
    else:
        #Materal == 5 is the voxel of the desired targer, we set it as brain
        MaterialMap[MaterialMap>4]=4
        BrainID=4
        LimSoft=4

    pAmpBrain=pAmp.copy()
    if 'MaterialMapCT' in Input:
        pAmpBrain[MaterialMap!=2]=0.0
    else:
        pAmpBrain[MaterialMap<4]=0.0

    # PressureTarget=pAmpBrain.max()
    # cx,cy,cz=np.where(pAmpBrain==PressureTarget)
    # cx=cx[0]
    # cy=cy[0]
    # cz=cz[0]
    cx=LocIJK[0]
    cy=LocIJK[1]
    cz=LocIJK[2]
    PressureTarget=pAmpBrain[cx,cy,cz]
    zl=cz
    
    print(' Max Pessure',cx,cy,cz,'\n',
          xf[cx],yf[cy],zf[cz],
          xf.shape,yf.shape,zf.shape,pAmp.shape,'\n',
          PressureTarget/1e6)

    
    
    PlanAtMaximum=pAmpBrain[:,:,cz]
    AcousticEnergy=(PlanAtMaximum**2/2/MaterialList['Density'][BrainID]/ MaterialList['SoS'][BrainID]*((xf[1]-xf[0])**2)).sum()
    print('Acoustic Energy at maximum plane',AcousticEnergy)
    
    if bCalculateLosses:
        assert(type(InputPData) is str) # we only do this for single focus
        WaterInputPData=InputPData.replace('DataForSim.h5','Water_DataForSim.h5')
        print('Load water',WaterInputPData)
        InputWater=ReadFromH5py(WaterInputPData)

        MateriaMapTissue=np.ascontiguousarray(np.flip(Input['MaterialMap'],axis=2))
        xfr=Input['x_vec']
        yfr=Input['y_vec']
        zfr=Input['z_vec']
        
        pAmpWater=np.ascontiguousarray(np.flip(InputWater['p_amp'],axis=2))
        PlanAtMaximumWater=pAmpWater[:,:,2] 
        AcousticEnergyWater=(PlanAtMaximumWater**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
        print('Water Acoustic Energy entering',AcousticEnergyWater)
        if 'MaterialMapCT' in Input:
            pAmpWater[MaterialMap!=2]=0.0
        else:
            pAmpWater[MaterialMap!=4]=0.0
        cxw,cyw,czw=np.where(pAmpWater==pAmpWater.max())
        cxw=cxw[0]
        cyw=cyw[0]
        czw=czw[0]
        print('Location Max Pessure Water',cxw,cyw,czw,'\n',
              xf[cxw],yf[cyw],zf[czw],pAmpWater.max()/1e6)
        
        pAmpTissue=np.ascontiguousarray(np.flip(Input['p_amp'],axis=2))
        if 'MaterialMapCT' in Input:
            pAmpTissue[MaterialMap!=2]=0.0
        else:
            pAmpTissue[MaterialMap!=4]=0.0

        cxr,cyr,czr=np.where(pAmpTissue==pAmpTissue.max())
        cxr=cxr[0]
        cyr=cyr[0]
        czr=czr[0]
        print('Location Max Pressure Tissue',cxr,cyr,czr,'\n',
              xfr[cxr],yfr[cyr],zfr[czr],pAmpTissue.max()/1e6)
        

        PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czw]
        AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
        print('Water Acoustic Energy at maximum plane water max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
        
        PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czr]
        AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
        print('Water Acoustic Energy at maximum plane tissue max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
        
        
        PlanAtMaximumTissue=pAmpTissue[:,:,czr] 
        AcousticEnergyTissue=(PlanAtMaximumTissue**2/2/MaterialList['Density'][BrainID]/ MaterialList['SoS'][BrainID]*((xf[1]-xf[0])**2)).sum()
        print('Tissue Acoustic Energy at maximum plane tissue',AcousticEnergyTissue)
        
        RatioLosses=AcousticEnergyTissue/AcousticEnergyWaterMaxLoc
        print('Total losses ratio and in dB',RatioLosses,np.log10(RatioLosses)*10)
        
    

    IntensityTarget=PressureTarget**2/(2.0*SaveDict['MaterialList']['SoS'][BrainID]*SaveDict['MaterialList']['Density'][BrainID])
    IntensityTarget=IntensityTarget/1e4
    print('IntensityTarget',IntensityTarget,SaveDict['MaterialList']['SoS'][BrainID],SaveDict['MaterialList']['Density'][BrainID])
    IntensityRatio=Isppa/IntensityTarget
    PressureAdjust=np.sqrt(Isppa*1e4*2.0*SaveDict['MaterialList']['SoS'][BrainID]*SaveDict['MaterialList']['Density'][BrainID])
    #PressureRatio=np.sqrt(IntensityRatio)
    PressureRatio=PressureAdjust/PressureTarget
    print('IntensityRatio,PressureRatio',IntensityRatio,PressureRatio)

    if 'MaterialMapCT' in Input:
        SelBrain=MaterialMap==2
    else:
        SelBrain=MaterialMap>=4

    SelSkin=MaterialMap==1
    if 'MaterialMapCT' in Input:
        SelSkull =MaterialMap>=3
    else:
        SelSkull =(MaterialMap>1) &\
            (MaterialMap<4)

    ##We calculate first for 1s, to find the hottest locations in each region
    TotalDurationSteps1s=int(1.0/dt)
    nStepsOn1s=TotalDurationSteps1s
    ResTemp,ResDose,MonitorSlice,Qarr=BHTE(pAmp*PressureRatio,
                                                      MaterialMap,
                                                      MaterialList,
                                                      (Input['x_vec'][1]-Input['x_vec'][0]),
                                                      TotalDurationSteps,
                                                      nStepsOn,
                                                      cy,
                                                      nFactorMonitoring=nFactorMonitoring,
                                                      dt=dt,
                                                      DutyCycle=DutyCycle,
                                                      Backend=Backend)

    ResTempSkin=ResTemp * SelSkin.astype(np.float32)
    ResTempBrain=ResTemp * SelBrain.astype(np.float32)
    ResTempSkull=ResTemp * SelSkull.astype(np.float32)
    mxSkin,mySkin,mzSkin=np.unravel_index(np.argmax(ResTempSkin, axis=None), ResTempSkin.shape)
    mxBrain,myBrain,mzBrain=np.unravel_index(np.argmax(ResTempBrain, axis=None), ResTempBrain.shape)
    mxSkull,mySkull,mzSkull=np.unravel_index(np.argmax(ResTempSkull, axis=None), ResTempSkull.shape)

    MonitoringPointsMap=np.zeros(MaterialMap.shape,np.uint32)
    MonitoringPointsMap[mxSkin,mySkin,mzSkin]=1
    MonitoringPointsMap[mxBrain,myBrain,mzBrain]=2
    MonitoringPointsMap[mxSkull,mySkull,mzSkull]=3
    if not(cx==mxBrain and cy==myBrain and cz==mzBrain):
        MonitoringPointsMap[cx,cy,cz]=4
   

    ResTemp,ResDose,MonitorSlice,Qarr,TemperaturePointsOn=BHTE(pAmp*PressureRatio,
                                                      MaterialMap,
                                                      MaterialList,
                                                      (Input['x_vec'][1]-Input['x_vec'][0]),
                                                      TotalDurationSteps,
                                                      nStepsOn,
                                                      cy,
                                                      nFactorMonitoring=nFactorMonitoring,
                                                      dt=dt,
                                                      DutyCycle=DutyCycle,
                                                      Backend=Backend,
                                                      MonitoringPointsMap=MonitoringPointsMap)

    FinalTemp,FinalDose,MonitorSliceOff,dum,TemperaturePointsOff=BHTE(pAmp*0,
                                                      MaterialMap,
                                                      MaterialList,
                                                      (Input['x_vec'][1]-Input['x_vec'][0]),
                                                      TotalDurationStepsOff,
                                                      0,
                                                      cy,
                                                      nFactorMonitoring=nFactorMonitoring,
                                                      dt=dt,
                                                      DutyCycle=DutyCycle,
                                                      Backend=Backend,
                                                      initT0=ResTemp,
                                                      initDose=ResDose,
                                                      MonitoringPointsMap=MonitoringPointsMap)
    
    TemperaturePoints=np.hstack((TemperaturePointsOn,TemperaturePointsOff))

    SaveDict['MonitorSlice']=MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1]
    SaveDict['mSkin']=np.array([mxSkin,mySkin,mzSkin]).astype(int)
    SaveDict['mBrain']=np.array([mxBrain,myBrain,mzBrain]).astype(int)
    SaveDict['mSkull']=np.array([mxSkull,mySkull,mzSkull]).astype(int)
    SaveDict['dt']=dt
    SaveDict['p_map']=pAmp*PressureRatio
    SaveDict['p_map_central']=pAmp[:,cy,:]*PressureRatio
    SaveDict['MaterialMap_central']=MaterialMap[:,cy,:]
    SaveDict['MaterialMap']=MaterialMap

    TI=ResTemp[SelBrain].max()
    
    TIS=ResTemp[SelSkin].max()

    TIC=ResTemp[SelSkull].max()
    
    print('Max. Temp. Brain, Max Temp. Skin, Max Temp. Skull',TI,TIS,TIC);

    CEMBrain=FinalDose[SelBrain].max()/60 # in min
    
    CEMSkin=FinalDose[SelSkin].max()/60 # in min

    CEMSkull=FinalDose[SelSkull].max()/60 # in min
    
    print('CEMBrain,CEMSkin,CEMSkull',CEMBrain,CEMSkin,CEMSkull)

    if 'MaterialMapCT' in Input:
        MaxBrainPressure = SaveDict['p_map'][SaveDict['MaterialMap']==3].max()
    else:
        MaxBrainPressure = SaveDict['p_map'][SaveDict['MaterialMap']==4].max()
        
    MI=MaxBrainPressure/1e6/np.sqrt(0.7)
    MaxIsppa=MaxBrainPressure**2/(2.0*SaveDict['MaterialList']['SoS'][BrainID]*SaveDict['MaterialList']['Density'][BrainID])
    MaxIsppa=MaxIsppa/1e4
    MaxIspta=DutyCycle*MaxIsppa

    Ispta =DutyCycle*Isppa

    SaveDict['MaxBrainPressure']=MaxBrainPressure
    if cx==mxBrain and cy==myBrain and cz==mzBrain:
        IndTarget=2
    else:
        IndTarget=3
    SaveDict['TempProfileTarget']=TemperaturePoints[IndTarget,:]
    SaveDict['TimeProfileTarget']=np.arange(SaveDict['TempProfileTarget'].size)*dt
    SaveDict['TemperaturePoints']=TemperaturePoints[:3,:] #these are max points in skin, brain and skull
    SaveDict['MI']=MI
    SaveDict['x_vec']=xf*1e3
    SaveDict['y_vec']=yf*1e3
    SaveDict['z_vec']=zf*1e3
    SaveDict['TI']=TI-37.0
    SaveDict['TIC']=TIC-37.0
    SaveDict['TIS']=TIS-37.0
    SaveDict['CEMBrain']=CEMBrain
    SaveDict['CEMSkin']=CEMSkin
    SaveDict['CEMSkull']=CEMSkull
    SaveDict['MaxIsppa']=MaxIsppa
    SaveDict['MaxIspta']=MaxIspta
    SaveDict['Isppa']=Isppa
    SaveDict['Ispta']=Ispta
    SaveDict['TempEndFUS']=ResTemp
    SaveDict['DoseEndFUS']=ResDose
    SaveDict['FinalTemp']=FinalTemp
    SaveDict['FinalDose']=FinalDose
    #we carry over these params to simplify analysis later
    ListOptionalParamsToCarryOver = ['ZSteering','YSteering','XSteering','RotationZ',
                                    'bDoRefocusing','DistanceConeToFocus','BasePhasedArrayProgramming',
                                    'BasePhasedArrayProgrammingRefocusing']
    for k in ListOptionalParamsToCarryOver:
        if k in Input:
            SaveDict[k]=Input[k]
        

    SaveDict['AdjustmentInRAS']=Input['AdjustmentInRAS']
    SaveDict['DistanceFromSkin']=Input['DistanceFromSkin']
    SaveDict['TxMechanicalAdjustmentZ']=Input['TxMechanicalAdjustmentZ']
    SaveDict['TargetLocation']=Input['TargetLocation']
    
    if bCalculateLosses:
        SaveDict['RatioLosses']=RatioLosses
    
    SaveToH5py(SaveDict,outfname+'.h5')
    savemat(outfname+'.mat',SaveDict)
    
    return outfname
        
