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

def GetThermalOutName(InputPData,DurationUS,DutyCycle,Isppa,PRF):
    return InputPData.split('.h5')[0]+'-ThermalField-Duration-%i-DC-%i-Isppa-%2.1fW-PRF-%iHz' % (DurationUS,DutyCycle*1000,Isppa,PRF)


def CalculateTemperatureEffects(InputPData,
                                DutyCycle=0.3,
                                Isppa=5,
                                sel_p='p_amp',
                                bPlot=True,
                                PRF=1500,
                                DurationUS=40,
                                bForceRecalc=False,
                                OutTemperature=37,
                                bCalculateLosses=False,
                                Backend='CUDA'):#this will help to calculate the final voltage to apply during experiments

    
    outfname=GetThermalOutName(InputPData,DurationUS,DutyCycle,Isppa,PRF)

    print(outfname)
    if bForceRecalc==False:
        if isfile(outfname+'.h5'):
            print('skipping', outfname)
            return outfname
    dt=0.01
    Input=ReadFromH5py(InputPData)
    
    #savemat(InputPData.split('.h5')[0]+'.mat',Input)
    MaterialList={}
    MaterialList['Density']=Input['Material'][:,0]
    MaterialList['SoS']=Input['Material'][:,1]
    MaterialList['Attenuation']=Input['Material'][:,3]
    #Water, Skin, Cortical, Trabecular, Brain

    #https://itis.swiss/virtual-population/tissue-properties/database/heat-capacity/
    MaterialList['SpecificHeat']=[4178,3391,1313,2274,3630] #(J/kg/°C)
    #https://itis.swiss/virtual-population/tissue-properties/database/thermal-conductivity/
    MaterialList['Conductivity']=[0.6,0.37,0.32,0.31,0.51] # (W/m/°C)
    #https://itis.swiss/virtual-population/tissue-properties/database/heat-transfer-rate/
    MaterialList['Perfusion']=np.array([0,106,10,30,559])
    
    MaterialList['Absorption']=np.array([0,0.85,0.16,0.15,0.85])

    MaterialList['InitTemperature']=[OutTemperature,37,37,37,37]
    SaveDict={}
    SaveDict['MaterialList']=MaterialList

    
    nFactorMonitoring=int(50e-3/dt) # we just track every 50 ms
    TotalDurationSteps=int((DurationUS+.001)/dt)
    nStepsOn=int(DurationUS/dt) 

    xf=Input['x_vec']
    yf=Input['y_vec']
    zf=Input['z_vec']
    
    pAmp=np.ascontiguousarray(np.flip(Input[sel_p],axis=2))
    print('pAmp.shape',pAmp.shape)
    
    MaterialMap=np.ascontiguousarray(np.flip(Input['MaterialMap'],axis=2))
    
    LocIJK=Input['TargetLocation'].flatten()
    
    #Materal == 5 is the voxel of the desired targer, we set it as brain
    MaterialMap[MaterialMap>4]=4
    
    pAmpBrain=pAmp.copy()
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
    AcousticEnergy=(PlanAtMaximum**2/2/MaterialList['Density'][4]/ MaterialList['SoS'][4]*((xf[1]-xf[0])**2)).sum()
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
        
        pAmpWater[MaterialMap!=4]=0.0
        cxw,cyw,czw=np.where(pAmpWater==pAmpWater.max())
        cxw=cxw[0]
        cyw=cyw[0]
        czw=czw[0]
        print('Location Max Pessure Water',cxw,cyw,czw,'\n',
              xf[cxw],yf[cyw],zf[czw],pAmpWater.max()/1e6)
        
        pAmpTissue=np.ascontiguousarray(np.flip(Input['p_amp'],axis=2))
        pAmpTissue[MateriaMapTissue<4]=0.0
        cxr,cyr,czr=np.where(pAmpTissue==pAmpTissue.max())
        cxr=cxr[0]
        cyr=cyr[0]
        czr=czr[0]
        print('Location Max Pessure Tissue',cxr,cyr,czr,'\n',
              xfr[cxr],yfr[cyr],zfr[czr],pAmpTissue.max()/1e6)
        
        PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czw]
        AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
        print('Water Acoustic Energy at maximum plane water max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
        
        PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czr]
        AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
        print('Water Acoustic Energy at maximum plane tissue max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
        
        
        PlanAtMaximumTissue=pAmpTissue[:,:,czr] 
        AcousticEnergyTissue=(PlanAtMaximumTissue**2/2/MaterialList['Density'][4]/ MaterialList['SoS'][4]*((xf[1]-xf[0])**2)).sum()
        print('Tissue Acoustic Energy at maximum plane tissue',AcousticEnergyTissue)
        
        RatioLosses=AcousticEnergyTissue/AcousticEnergyWaterMaxLoc
        print('Total losses ratio and in dB',RatioLosses,np.log10(RatioLosses)*10)
        
    

    IntensityTarget=PressureTarget**2/(2.0*SaveDict['MaterialList']['SoS'][4]*SaveDict['MaterialList']['Density'][4])
    IntensityTarget=IntensityTarget/1e4
    IntensityRatio=Isppa/IntensityTarget
    PressureAdjust=np.sqrt(Isppa*1e4*2.0*SaveDict['MaterialList']['SoS'][4]*SaveDict['MaterialList']['Density'][4])
    #PressureRatio=np.sqrt(IntensityRatio)
    PressureRatio=PressureAdjust/PressureTarget
    print('IntensityRatio,PressureRatio',IntensityRatio,PressureRatio)
    
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
    

    #SaveDict['ResTemp']=ResTemp
    SaveDict['MonitorSlice']=MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1]
    
    SaveDict['p_map']=pAmp[:,cy,:].copy()*PressureRatio
    SaveDict['MaterialMap']=MaterialMap[:,cy,:]
    
    SelBrain=MaterialMap[:,cy,:]==4

    SelSkin = MaterialMap[:,cy,:]==1

    SelSkull = (MaterialMap[:,cy,:]>1) &\
                (MaterialMap[:,cy,:]<4)

    TI=(MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1][SelBrain]).max()

    TIS=(MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1][SelSkin]).max()

    TIC=(MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1][SelSkull]).max()
    
    print('TI,TIS,TIC',TI-37,TIS-37,TIC-37);

    MaxBrainPressure = SaveDict['p_map'][SaveDict['MaterialMap']==4].max()
    MI=MaxBrainPressure/1e6/np.sqrt(0.7)
    MaxIsppa=MaxBrainPressure**2/(2.0*SaveDict['MaterialList']['SoS'][4]*SaveDict['MaterialList']['Density'][4])
    MaxIsppa=MaxIsppa/1e4
    MaxIspta=DutyCycle*MaxIsppa

    Ispta =DutyCycle*Isppa

    SaveDict['MaxBrainPressure']=MaxBrainPressure
    SaveDict['TempProfileTarget']=MonitorSlice[cy,zl,:]
    SaveDict['TimeProfileTarget']=np.arange(SaveDict['TempProfileTarget'].size)*dt*nFactorMonitoring;
    SaveDict['MI']=MI
    SaveDict['x_vec']=xf*1e3
    SaveDict['y_vec']=yf*1e3
    SaveDict['z_vec']=zf*1e3
    SaveDict['TI']=TI-37.0
    SaveDict['TIC']=TIC-37.0
    SaveDict['TIS']=TIS-37.0
    SaveDict['MaxIsppa']=MaxIsppa
    SaveDict['MaxIspta']=MaxIspta
    SaveDict['Isppa']=Isppa
    SaveDict['Ispta']=Ispta
    SaveDict['TempEndFUS']=ResTemp
    SaveDict['DoseEndFUS']=ResDose
    #we carry over these params to simplify analysis later
    SaveDict['ZSteering']=Input['ZSteering']
    SaveDict['AdjustmentInRAS']=Input['AdjustmentInRAS']
    SaveDict['DistanceFromSkin']=Input['DistanceFromSkin']
    SaveDict['TxMechanicalAdjustmentZ']=Input['TxMechanicalAdjustmentZ']
    SaveDict['TargetLocation']=Input['TargetLocation'][[0,2]]
    
    if bCalculateLosses:
        SaveDict['RatioLosses']=RatioLosses
    
    SaveToH5py(SaveDict,outfname+'.h5')
    savemat(outfname+'.mat',SaveDict)
    
    if bPlot:
 
        AllContours=[]
        #skin
        contours,_ = cv.findContours((SaveDict['MaterialMap']==1).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        AllContours.append(contours)   
        #skull
        contours,_ = cv.findContours(((SaveDict['MaterialMap']==2)|(SaveDict['MaterialMap']==3)).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        AllContours.append(contours)  
                
        plt.figure(figsize=(12,6))
        ax=plt.subplot(1,2,1)
        plt.imshow(SaveDict['p_map'].T,extent=[xf.min()*1e3,xf.max()*1e3,zf.max()*1e3,zf.min()*1e3],
                   cmap=plt.cm.jet)
        plt.colorbar()
        #we add contours of skin and skull bone
        sr=['y-','w-']
        for n in range(2):
            contours=AllContours[n]
            for c in contours:
                ax.plot(SaveDict['x_vec'][c[:,0,1]],SaveDict['z_vec'][c[:,0,0]],sr[n],linewidth=1)
            

        ax=plt.subplot(1,2,2)
        plt.imshow(MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1].T,
                   extent=[xf.min()*1e3,xf.max()*1e3,zf.max()*1e3,zf.min()*1e3],cmap=plt.cm.jet,vmin=OutTemperature)
        plt.colorbar()
        for n in range(2):
            contours=AllContours[n]
            for c in contours:
                ax.plot(SaveDict['x_vec'][c[:,0,1]],SaveDict['z_vec'][c[:,0,0]],sr[n],linewidth=1)
            
        plt.suptitle(outfname.split('/')[-1])
        plt.figure(figsize=(8,4))
        plt.plot(SaveDict['TimeProfileTarget'],SaveDict['TempProfileTarget'])
        plt.title(outfname.split('/')[-1])
    
    return outfname
        
