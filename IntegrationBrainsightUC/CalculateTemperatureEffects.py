'''
Tools to calculate thermal simulations for LIFU experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 23, 2021
     last update   - Nov 27, 2021

'''

import numpy as np
import matplotlib.pyplot as plt
from  scipy.io import loadmat,savemat
from BabelViscoFDTD.tools.RayleighAndBHTE import BHTE,BHTEMultiplePressureFields
from BabelViscoFDTD.H5pySimple import SaveToH5py,ReadFromH5py
from scipy.io import loadmat,savemat
from platform import platform
from os.path import isfile

def CalculateTemperatureEffects(InputPData,DutyCycle,Isppa,sel_p='p_amp',
                                bPlot=True,
                                PRF=1500,
                                bUseTargetLocations=False,
                                bForceRecalc=False):
    if 'Windows' in platform() or 'Linux' in platform():
        Backend='CUDA'
    else:
        Backend ='OpenCL'
    
    if type(InputPData) is str:
        # assert('TxMoved_DataForSim.h5' in InputPData)
        outfname=InputPData.split('.h5')[0]+'-ThermalField-DC-%i-Isppa-%2.1fW' % (DutyCycle*1000,Isppa)
        
    else:
        assert('TxMoved' in InputPData[0])
        outfname=InputPData[0].split('.h5')[0]+'-ThermalField-DC-%i-Isppa-%2.1fW-PRF-%iHz' % (DutyCycle*1000,Isppa,PRF)
        
    print(outfname)
    if bForceRecalc==False:
        if isfile(outfname+'.h5'):
            print('skipping', outfname)
            return outfname
    if type(InputPData) is str:
        dt=0.01
        Input=ReadFromH5py(InputPData)
    else:
        dt=0.01
        
        #this has to be always the center steering point!
        Input=ReadFromH5py(InputPData[0])
        
        AllInputs=np.zeros((len(InputPData),Input[sel_p].shape[0],Input[sel_p].shape[1],
                            Input[sel_p].shape[2]),Input[sel_p].dtype)
        AllInputs[0,:,:,:]=np.ascontiguousarray(np.flip(Input[sel_p],axis=2)).copy()
        for n in range(len(InputPData)):
            AllInputs[n,:,:,:]=np.ascontiguousarray(np.flip(ReadFromH5py(InputPData[n])[sel_p],axis=2))
        NCyclesOn=int(DutyCycle/dt)
        NCyclesOff=int(1.0/dt)-NCyclesOn
        assert(NCyclesOn>0)
        assert(NCyclesOff>0 )
        
        nStepsOnOffList=np.zeros((len(InputPData),2),np.int32)
        #all points have the same duration
        nStepsOnOffList[:,0]=NCyclesOn
        nStepsOnOffList[:,1]=NCyclesOff
            
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

    MaterialList['InitTemperature']=[37,37,37,37,37]
    SaveDict={}
    SaveDict['MaterialList']=MaterialList

    
    nFactorMonitoring=int(50e-3/dt) # we just track every 50 ms
    if type(InputPData) is str:
        DurationUS=40
    else:
        DurationUS=120
    TotalDurationSteps=int((DurationUS+.001)/dt)
    nStepsOn=int(DurationUS/dt) 

    xf=Input['x_vec']
    yf=Input['y_vec']
    zf=Input['z_vec']
    
    pAmp=np.ascontiguousarray(np.flip(Input[sel_p],axis=2))
    print('pAmp.shape',pAmp.shape)
    
    MaterialMap=np.ascontiguousarray(np.flip(Input['MaterialMap'],axis=2))
    
    pAmpBrain=pAmp.copy()
    pAmpBrain[MaterialMap!=4]=0.0
    
#     cx=np.argmin(np.abs(xf))
#     cy=np.argmin(np.abs(yf))
#     cz=np.argmin(np.abs(zf-0.135))
#     zl=cz
    
    #PressureTarget=pAmp[cx,cy,zl]
    PressureTarget=pAmpBrain.max()
    if bUseTargetLocations:
        cx=Input['TargetLocation'][0]
        cy=Input['TargetLocation'][1]
        cz=Input['TargetLocation'][2]
    else:
        cx,cy,cz=np.where(pAmpBrain==PressureTarget)
        cx=cx[0]
        cy=cy[0]
        cz=cz[0]
    zl=cz
    print('Location Max Pessure',cx,cy,cz,'\n',
          xf[cx],yf[cy],zf[cz],
          xf.shape,yf.shape,zf.shape,pAmp.shape,'\n',
          PressureTarget/1e6)
    
    
    PlanAtMaximum=pAmpBrain[:,:,cz]
    AcousticEnergy=(PlanAtMaximum**2/2/MaterialList['Density'][4]/ MaterialList['SoS'][4]*((xf[1]-xf[0])**2)).sum()
    
    print('Acoustic Energy at maximum plane',AcousticEnergy)
    

    assert(type(InputPData) is str) # we only do this for single focus
    WaterInputPData=InputPData.replace('DataForSim.h5','WaterOnly_DataForSim.h5')
    InputWater=ReadFromH5py(WaterInputPData)
    
    RefocuInputPData=InputPData.replace('TxMoved_DataForSim.h5','DataForSim.h5')
    InputRefocus=ReadFromH5py(RefocuInputPData)
    #the refocus case will have a slighlty different matrix dimensions
    MateriaMapRefocus=np.ascontiguousarray(np.flip(InputRefocus['MaterialMap'],axis=2))
    xfr=InputRefocus['x_vec']
    yfr=InputRefocus['y_vec']
    zfr=InputRefocus['z_vec']
    
    pAmpWater=np.ascontiguousarray(np.flip(InputWater['p_amp'],axis=2))
    pAmpWater[MaterialMap!=4]=0.0
    if bUseTargetLocations:
        cxw=InputRefocus['TargetLocation'][0]
        cyw=InputRefocus['TargetLocation'][1]
        czw=InputRefocus['TargetLocation'][2]
    else:
        cxw,cyw,czw=np.where(pAmpWater==pAmpWater.max())
        cxw=cxw[0]
        cyw=cyw[0]
        czw=czw[0]
    print('Location Max Pessure Water',cxw,cyw,czw,'\n',
        xf[cxw],yf[cyw],zf[czw],pAmpWater.max()/1e6)
    
    pAmpRefocus=np.ascontiguousarray(np.flip(InputRefocus['p_amp_refocus'],axis=2))
    pAmpRefocus[MateriaMapRefocus!=4]=0.0
    cxr,cyr,czr=np.where(pAmpRefocus==pAmpRefocus.max())
    cxr=cxr[0]
    cyr=cyr[0]
    czr=czr[0]
    print('Location Max Pessure Refocus',cxr,cyr,czr,'\n',
        xfr[cxr],yfr[cyr],zfr[czr],pAmpRefocus.max()/1e6)
    
    PlanAtMaximumWater=pAmpWater[:,:,cz] 
    AcousticEnergyWater=(PlanAtMaximumWater**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Acoustic Energy at maximum plane water',AcousticEnergyWater)
    
    PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czw]
    AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Acoustic Energy at maximum plane water max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
    
    PlanAtMaximumRefocus=pAmpRefocus[:,:,czr] 
    AcousticEnergyRefocus=(PlanAtMaximumRefocus**2/2/MaterialList['Density'][4]/ MaterialList['SoS'][4]*((xf[1]-xf[0])**2)).sum()
    print('Acoustic Energy at maximum plane refocus',AcousticEnergyRefocus)
    if sel_p=='p_amp':
        RatioLosses=AcousticEnergyRefocus/AcousticEnergyWaterMaxLoc
    else:
        RatioLosses=AcousticEnergy/AcousticEnergyWaterMaxLoc
    print('Total losses ratio and in dB with',sel_p,RatioLosses,np.log10(RatioLosses)*10)
        
    
    PressureAdjust=np.sqrt(Isppa*1e4*2.0*SaveDict['MaterialList']['SoS'][BrainID]*SaveDict['MaterialList']['Density'][BrainID])
    PressureRatio=PressureAdjust/pAmpTissue.max()
    
    if type(InputPData) is str:
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
    else:
        ResTemp,ResDose,MonitorSlice,Qarr=BHTEMultiplePressureFields(AllInputs*PressureRatio,
                                                      MaterialMap,
                                                      MaterialList,
                                                      (Input['x_vec'][1]-Input['x_vec'][0]),
                                                      TotalDurationSteps,
                                                      nStepsOnOffList,
                                                      cy,
                                                      nFactorMonitoring=nFactorMonitoring,
                                                      dt=dt,
                                                      Backend=Backend)
        ##we combine all acoustic fields to show the coverage
        pAmp=np.max(AllInputs,axis=0)
        
        
       


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
    
    print('Max. Temp. Brain, Max Temp. Skin, Max Temp. Skull',TI,TIS,TIC);

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
    SaveDict['TargetLocation']=Input['TargetLocation'][[0,2]]
    
    SaveDict['RatioLosses']=RatioLosses
    
    SaveToH5py(SaveDict,outfname+'.h5')
    savemat(outfname+'.mat',SaveDict)
    
    if bPlot:
 
        plt.figure()
        plt.plot(zf,SaveDict['p_map'][cx,:])
        plt.figure(figsize=(12,12))
        plt.subplot(2,2,1)
        plt.imshow(MaterialMap[:,cy,:].T,
                   extent=[xf.min(),xf.max(),zf.max(),zf.min()],cmap=plt.cm.jet)
        plt.colorbar()

        plt.subplot(2,2,2)
        plt.imshow(SaveDict['p_map'].T,extent=[xf.min(),xf.max(),zf.max(),zf.min()],
                   cmap=plt.cm.jet)
        plt.colorbar()

        plt.subplot(2,2,3)
        plt.imshow(MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1].T,
                   extent=[xf.min(),xf.max(),zf.max(),zf.min()],cmap=plt.cm.jet)
        plt.colorbar()

        plt.subplot(2,2,4)
        plt.imshow((MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1] *
                   (MaterialMap[:,cy,:]==1).astype(float)).T,
                   extent=[xf.min(),xf.max(),zf.max(),zf.min()],cmap=plt.cm.jet)
        plt.colorbar()
        plt.figure()
        plt.plot(SaveDict['TimeProfileTarget'],SaveDict['TempProfileTarget'])
    
    return outfname
        