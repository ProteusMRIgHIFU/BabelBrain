'''
Tools to calculate thermal simulations for TUS experiments

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
from BabelViscoFDTD.tools.RayleighAndBHTE import  InitOpenCL, InitCuda, InitMetal
from multiprocessing import Process,Queue
import sys
import time

class InOutputWrapper(object):
    def __init__(self, queue, stdout=True):
        self.queue=queue
        if stdout:
            self._stream = sys.stdout
            sys.stdout = self
        else:
            self._stream = sys.stderr
            sys.stderr = self
        self._stdout = stdout

    def write(self, text):
        self.queue.put(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            if self._stdout:
                sys.stdout = self._stream
            else:
                sys.stderr = self._stream
        except AttributeError:
            pass

def GetThermalOutName(InputPData,DurationUS,DurationOff,DutyCycle,Isppa,PRF,Repetitions):
    if DurationUS>=1 and DurationOff>=1:
        suffix = '-ThermalField-Duration-%i-DurationOff-%i-DC-%i-Isppa-%2.1fW-PRF-%iHz' % (DurationUS,DurationOff,DutyCycle*1000,Isppa,PRF)
    else:
        suffix = '-ThermalField-Duration-%3.2f-DurationOff-%3.2f-DC-%i-Isppa-%2.1fW-PRF-%iHz' % (DurationUS,DurationOff,DutyCycle*1000,Isppa,PRF)
    if Repetitions >1:
        suffix+='-%iReps' % (Repetitions)
    if '__Steer_X' in InputPData:
        #we check if this a case for multifocal delivery
        return InputPData.split('__Steer_X')[0]+'_DataForSim'+suffix
    else:
        return InputPData.split('.h5')[0]+suffix

def AnalyzeLosses(pAmp,MaterialMap,LocIJK,Input,
                  MaterialList,pAmpWater,Isppa,
                  xf,yf,zf,SelBrain,bForceHomogenousMedium,bSegmentedBrain):
    pAmpBrain=pAmp.copy()

    SoSMap=MaterialList['SoS'][MaterialMap]
    DensityMap=MaterialList['Density'][MaterialMap]

    pAmpBrain[SelBrain==False]=0.0
    
    cz=LocIJK[2]
    
    PlanAtMaximum=pAmpBrain[:,:,cz]
    AcousticEnergy=(PlanAtMaximum**2/2/DensityMap[:,:,cz]/ SoSMap[:,:,cz]*((xf[1]-xf[0])**2)).sum()
    print('Acoustic Energy at maximum plane',AcousticEnergy)
    

    xfr=Input['x_vec']
    yfr=Input['y_vec']
    zfr=Input['z_vec'].copy()
    zfr-=zfr.min()
    
    
    PlanAtMaximumWater=pAmpWater[:,:,2] 
    AcousticEnergyWater=(PlanAtMaximumWater**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Water Acoustic Energy entering',AcousticEnergyWater)
    if not bForceHomogenousMedium:
        if bSegmentedBrain:
            if 'MaterialMapCT' in Input:
                selregion = np.isin(MaterialMap,[2,3,4,5])==False
            else:
                selregion = np.isin(MaterialMap,[0,1,2,3])
        else:
            if 'MaterialMapCT' in Input:
                selregion=MaterialMap!=2
            else:
                selregion=MaterialMap!=4
    pAmpWater[selregion]=0.0
    cxw,cyw,czw=np.where(pAmpWater==pAmpWater.max())
    cxw=cxw[0]
    cyw=cyw[0]
    czw=czw[0]
    print('Location Max Pessure Water',cxw,cyw,czw,'\n',
            xf[cxw],yf[cyw],zf[czw]-zf[0],pAmpWater.max()/1e6)
    
    pAmpTissue=np.ascontiguousarray(np.flip(Input['p_amp'],axis=2))
    pAmpTissue[SelBrain==False]=0.0

    cxr,cyr,czr=np.where(pAmpTissue==pAmpTissue.max())
    cxr=cxr[0]
    cyr=cyr[0]
    czr=czr[0]
    print('Location Max Pressure Tissue',cxr,cyr,czr,'\n',
            xfr[cxr],yfr[cyr],zfr[czr],pAmpTissue.max()/1e6)
    

    PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czw]
    AcousticEnergyWaterMaxLocWat=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Water Acoustic Energy at maximum plane water max loc',AcousticEnergyWaterMaxLocWat) #must be very close to AcousticEnergyWater

    PlanAtMaximumTissue=pAmpTissue[:,:,czw] 
    AcousticEnergyTissueMaxLocWat=(PlanAtMaximumTissue**2/2/DensityMap[:,:,czw]/SoSMap[:,:,czw]*((xf[1]-xf[0])**2)).sum()
    print('Tissue Acoustic Energy at maximum plane water max loc',AcousticEnergyTissueMaxLocWat)
    
    PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czr]
    AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Water Acoustic Energy at maximum plane tissue max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
    
    PlanAtMaximumTissue=pAmpTissue[:,:,czr] 
    AcousticEnergyTissue=(PlanAtMaximumTissue**2/2/DensityMap[:,:,czr]/ SoSMap[:,:,czr]*((xf[1]-xf[0])**2)).sum()
    print('Tissue Acoustic Energy at maximum plane tissue',AcousticEnergyTissue)
    
    RatioLossesLoc=AcousticEnergyTissueMaxLocWat/AcousticEnergyWaterMaxLocWat
    print('Total losses ratio using Water loc and in dB',RatioLossesLoc,np.log10(RatioLossesLoc)*10)

    RatioLosses=AcousticEnergyTissue/AcousticEnergyWaterMaxLoc
    print('Total losses ratio and in dB',RatioLosses,np.log10(RatioLosses)*10)

    if RatioLosses > (RatioLossesLoc+0.2):
        print('Warning: RatioLossesLoc is bigger than RatioLosses by more than 20%\nUsing water loc for ratio losses')
        RatioLosses=RatioLossesLoc

    if bSegmentedBrain:
        SoSTarget = SoSMap[cxr,cyr,czr]
        DensityTarget = DensityMap[cxr,cyr,czr]
    else:
        SoSTarget = SoSMap[LocIJK[0],LocIJK[1],LocIJK[2]]
        DensityTarget = DensityMap[LocIJK[0],LocIJK[1],LocIJK[2]]

    PressureAdjust=np.sqrt(Isppa*1e4*2.0*SoSTarget*DensityTarget)
    PressureRatio=PressureAdjust/pAmpTissue.max()
    return PressureRatio,RatioLosses


def RunBHTECycles(nCurrent,
                    Repetitions,
                    TotalIterations,
                    TotalDurationBetweenGroups,
                    TotalDurationStepsOff,
                    LimitBHTEIterationsPerProcess,
                    InputPData,
                    PMaps,
                    MaterialMap,
                    MaterialList,
                    dx,
                    TotalDurationSteps,
                    nStepsOn,
                    cy,
                    nFactorMonitoring,
                    dt,
                    DutyCycle,
                    Backend,
                    MonitoringPointsMap,
                    stableTemp,
                    TemperaturePoints,
                    FinalTemp,
                    FinalDose,
                    bRunInSubProcess=False,
                    ):
    if type(InputPData) is str:
        p0=PMaps*0
    else:
        p0=PMaps[0,:,:,:]*0
    for nCurrent in range(nCurrent,TotalIterations):
        if nCurrent >0 :
            initT0=FinalTemp
            initDose=FinalDose
        else:
            initT0=None
            initDose=None
            
        if type(InputPData) is str:
            ResTemp,ResDose,MonitorSlice,Qarr,TemperaturePointsOn=BHTE(PMaps,
                                                            MaterialMap,
                                                            MaterialList,
                                                            dx,
                                                            TotalDurationSteps,
                                                            nStepsOn,
                                                            cy,
                                                            nFactorMonitoring=nFactorMonitoring,
                                                            dt=dt,
                                                            DutyCycle=DutyCycle,
                                                            Backend=Backend,
                                                            initT0=initT0,
                                                            initDose=initDose,
                                                            MonitoringPointsMap=MonitoringPointsMap,
                                                            stableTemp=stableTemp)
        else:
            ResTemp,ResDose,MonitorSlice,Qarr,TemperaturePointsOn=BHTEMultiplePressureFields(PMaps,
                                                            MaterialMap,
                                                            MaterialList,
                                                            dx,
                                                            TotalDurationSteps,
                                                            nStepsOn,
                                                            cy,
                                                            nFactorMonitoring=nFactorMonitoring,
                                                            dt=dt,
                                                            Backend=Backend,
                                                            initT0=initT0,
                                                            initDose=initDose,
                                                            MonitoringPointsMap=MonitoringPointsMap,
                                                            stableTemp=stableTemp)
            
        
        #for cooling off, we do not need to do steering, just running with no energy
        FinalTemp,FinalDose,MonitorSliceOff,dum,TemperaturePointsOff=BHTE(p0,
                                                        MaterialMap,
                                                        MaterialList,
                                                        dx,
                                                        TotalDurationStepsOff,
                                                        0,
                                                        cy,
                                                        nFactorMonitoring=nFactorMonitoring,
                                                        dt=dt,
                                                        DutyCycle=DutyCycle,
                                                        Backend=Backend,
                                                        initT0=ResTemp,
                                                        initDose=ResDose,
                                                        MonitoringPointsMap=MonitoringPointsMap,
                                                        stableTemp=stableTemp) 
    
        if nCurrent==0:
            TemperaturePoints=np.hstack((TemperaturePointsOn,TemperaturePointsOff))
        else:
            TemperaturePoints=np.hstack((TemperaturePoints,TemperaturePointsOn,TemperaturePointsOff))

        print('nCurrent,TotalIterations',nCurrent,TotalIterations)

        if (nCurrent+1)%Repetitions == 0 and TotalDurationBetweenGroups>0.0:
            #we ran the extra time off pause
            FinalTemp,FinalDose,MonitorSliceOff,dum,TemperaturePointsOff=BHTE(p0,
                                                            MaterialMap,
                                                            MaterialList,
                                                            dx,
                                                            TotalDurationBetweenGroups,
                                                            0,
                                                            cy,
                                                            nFactorMonitoring=nFactorMonitoring,
                                                            dt=dt,
                                                            DutyCycle=DutyCycle,
                                                            Backend=Backend,
                                                            initT0=FinalTemp,
                                                            initDose=FinalDose,
                                                            MonitoringPointsMap=MonitoringPointsMap,
                                                            stableTemp=stableTemp)
            TemperaturePoints=np.hstack((TemperaturePoints,TemperaturePointsOff))
        if (bRunInSubProcess) and\
            ((nCurrent+1)% LimitBHTEIterationsPerProcess==0 or (nCurrent+1)==TotalIterations):
            print('Finishing sub process')
            break
    return ResTemp,ResDose,FinalTemp,FinalDose,TemperaturePoints,nCurrent+1

def RunInProcess(queueResult,Backend,deviceName,queueMsg,
                 LimitBHTEIterationsPerProcess,nCurrent,
                 NumberGroupedSonications,Repetitions,
                 InputPData,PMaps,MaterialMap,
                 MaterialList,dx,TotalDurationSteps,
                 TotalDurationStepsOff,
                 nStepsOn,cy,nFactorMonitoring,dt,
                 DutyCycle,MonitoringPointsMap,
                 TemperaturePoints,stableTemp,
                 FinalTemp,FinalDose,
                 TotalDurationBetweenGroups):
    stdout = InOutputWrapper(queueMsg,True)
    
    if Backend=='CUDA':
        InitCuda(deviceName)
    elif Backend=='OpenCL':
        InitOpenCL(deviceName)
    else:
        InitMetal(deviceName)

    
    TotalIterations=NumberGroupedSonications*Repetitions

    Res=RunBHTECycles(nCurrent,
                    Repetitions,
                    TotalIterations,
                    TotalDurationBetweenGroups,
                    TotalDurationStepsOff,
                    LimitBHTEIterationsPerProcess,
                    InputPData,
                    PMaps,
                    MaterialMap,
                    MaterialList,
                    dx,
                    TotalDurationSteps,
                    nStepsOn,
                    cy,
                    nFactorMonitoring,
                    dt,
                    DutyCycle,
                    Backend,
                    MonitoringPointsMap,
                    stableTemp,
                    TemperaturePoints,
                    FinalTemp,
                    FinalDose,
                    bRunInSubProcess=True,
                    )
    queueResult.put(Res)
    

def CalculateTemperatureEffects(InputPData,
                                deviceName,
                                queueMsg,
                                DutyCycle=0.3,
                                Isppa=5,
                                sel_p='p_amp',
                                PRF=1500,
                                DurationUS=40,
                                DurationOff=40,
                                Repetitions=1,
                                bForceRecalc=False,
                                BaselineTemperature=37,
                                bGlobalDCMultipoint=False,
                                Frequency=7e5,
                                NumberGroupedSonications=1,
                                PauseBetweenGroupedSonications=0.0,
                                Backend='CUDA',
                                LimitBHTEIterationsPerProcess=100,
                                bForceHomogenousMedium=False,
                                bForceNoAbsorptionSkullScalp=False,
                                HomogenousMediumValues={'ThermalConductivity':0.5,  
                                    'SpecificHeat':3583.0,
                                    'Perfusion':55.0,
                                    'Absorption':0.85, #m/s
                                    'InitTemperature':37.0}, #Np/m
                                ):


    if type(InputPData) is str:    
        outfname=GetThermalOutName(InputPData,DurationUS,DurationOff,DutyCycle,Isppa,PRF,Repetitions)
    else:
        outfname=GetThermalOutName(InputPData[0],DurationUS,DurationOff,DutyCycle,Isppa,PRF,Repetitions)
        if len(InputPData)==1:
            #we simplify if we only have a single file in the list
            InputPData=InputPData[0]
    print('InputPData',InputPData)
    print(outfname)

    print('Thermal sim with Backend',Backend)
    print('Operating Frequency',Frequency)
    print('Baseline Temperature',BaselineTemperature)
    if bForceRecalc==False:
        if isfile(outfname+'.h5'):
            print('skipping', outfname)
            return outfname
    dt=0.01
    
    if type(InputPData) is str:   
        Input=ReadFromH5py(InputPData)
        WaterInputPData=InputPData.replace('DataForSim.h5','Water_DataForSim.h5')
        print('Load water',WaterInputPData)
        InputWater=ReadFromH5py(WaterInputPData)
        
        pAmp=np.ascontiguousarray(np.flip(Input[sel_p],axis=2))
        pAmpWater=np.ascontiguousarray(np.flip(InputWater['p_amp'],axis=2))
        
    else:
        ALL_ACFIELDSKULL=[]
        Input=ReadFromH5py(InputPData[0])
        
        AllInputs=np.zeros((len(InputPData),Input[sel_p].shape[0],Input[sel_p].shape[1],
                            Input[sel_p].shape[2]),Input[sel_p].dtype)
        AllInputsWater=np.zeros((len(InputPData),Input[sel_p].shape[0],Input[sel_p].shape[1],
                            Input[sel_p].shape[2]),Input[sel_p].dtype)
        for n in range(len(InputPData)):
            ALL_ACFIELDSKULL.append(ReadFromH5py(InputPData[n]))
            AllInputs[n,:,:,:]=np.ascontiguousarray(np.flip(ALL_ACFIELDSKULL[-1][sel_p],axis=2))
            fwater=InputPData[n].replace('DataForSim.h5','Water_DataForSim.h5')
            AllInputsWater[n,:,:,:]=np.ascontiguousarray(np.flip(ReadFromH5py(fwater)['p_amp'],axis=2))
        
        if DurationUS>len(InputPData)*2 and bGlobalDCMultipoint: 
        #ad-hoc rule, if sonication last at least 2x seconds the number of focal spots, we  approximate the heating as each point would take 1 second (with DC indicating how much percentage will  be on), this is valid for long sonications
            DurationCalculations=1.0 
        else:
            DurationCalculations=0.1 # for short sonications, we approximate in chunks of 0.1 seconds
        if bGlobalDCMultipoint:
            while(True): #we need to add a fix for low duty cycle
                NCyclesOn=int(DutyCycle*DurationCalculations/dt/len(InputPData))
                if NCyclesOn>0:
                    break
                DurationCalculations+=0.1
            NCyclesOff=int(DurationCalculations/dt/len(InputPData))-NCyclesOn
        else:
            while(True): #we need to add a fix for low duty cycle
                NCyclesOn=int(DutyCycle*DurationCalculations/dt)
                if NCyclesOn>0:
                    break
                DurationCalculations+=0.1
            NCyclesOff=int(DurationCalculations/dt)-NCyclesOn
        assert(NCyclesOn>0)
        assert(NCyclesOff>0 )
        
        nStepsOnOffList=np.zeros((len(InputPData),2),np.int32)
        #all points have the same duration
        nStepsOnOffList[:,0]=NCyclesOn
        nStepsOnOffList[:,1]=NCyclesOff
        
    if 'MaterialMapCT' in Input:
        MaterialMap=np.ascontiguousarray(np.flip(Input['MaterialMapCT'],axis=2))
        OrigMaterialMap=np.ascontiguousarray(np.flip(Input['MaterialMap'],axis=2))
    else:
        MaterialMap=np.ascontiguousarray(np.flip(Input['MaterialMap'],axis=2))
        OrigMaterialMap=MaterialMap

    bSegmentedBrain = np.any(OrigMaterialMap>5)
    
    MaterialList={}
    MaterialList['Density']=Input['Material'][:,0]
    MaterialList['SoS']=Input['Material'][:,1]
    MaterialList['Attenuation']=Input['Material'][:,3]
    if bForceNoAbsorptionSkullScalp:
        print('Forcing no absorption in Skull and Scalp')
    if bForceHomogenousMedium:
        print('Running BHTE with homogenous medium with ',HomogenousMediumValues)
        MaterialList['SpecificHeat']=np.array([4178.0,HomogenousMediumValues['SpecificHeat']]) #(J/kg/°C)
        MaterialList['Conductivity']=np.array([0.6,HomogenousMediumValues['ThermalConductivity']]) # (W/m/°C)
        MaterialList['Perfusion']=np.array([0.0,HomogenousMediumValues['Perfusion']])
        MaterialList['Absorption']=np.array([0,HomogenousMediumValues['Absorption']])
        MaterialList['InitTemperature']=np.array([HomogenousMediumValues['InitTemperature'],HomogenousMediumValues['InitTemperature']])
        BaselineTemperature=HomogenousMediumValues['InitTemperature']
    
    elif 'MaterialMapCT' not in Input:
        #Water, Skin, Cortical, Trabecular, Brain

        #https://itis.swiss/virtual-population/tissue-properties/database/heat-capacity/
        MaterialList['SpecificHeat']=[4178.0,3391.0,1313.0,2274.0,3630.0] #(J/kg/°C)
        #https://itis.swiss/virtual-population/tissue-properties/database/thermal-conductivity/
        MaterialList['Conductivity']=[0.6,0.37,0.32,0.31,0.51] # (W/m/°C)
        #https://itis.swiss/virtual-population/tissue-properties/database/heat-transfer-rate/
        MaterialList['Perfusion']=[0.0,106.0,10.0,30.0,559.0]
        MaterialList['Absorption']=[0,0.85,0.16,0.15,0.85]
        if bForceNoAbsorptionSkullScalp:
            MaterialList['Absorption'][1]=0.0
            MaterialList['Absorption'][2]=0.0
            MaterialList['Absorption'][3]=0.0
            
        MaterialList['InitTemperature']=[BaselineTemperature,BaselineTemperature,
                                         BaselineTemperature,BaselineTemperature,BaselineTemperature]
        if bSegmentedBrain:
            #we add white matter, gray matter and CSF
            MaterialList['Conductivity']+=[0.55, 0.48, 0.57]
            MaterialList['SpecificHeat']+=[3583.0, 3696.0, 4096.0]
            MaterialList['Perfusion']+=[764.0, 212.0, 0.0]
            MaterialList['Absorption']+=[0.85, 0.85, 0.0]
            MaterialList['InitTemperature']+=[BaselineTemperature,BaselineTemperature,BaselineTemperature]

        for k in ['SpecificHeat','Conductivity','Perfusion','Absorption','InitTemperature']:
            MaterialList[k]=np.array(MaterialList[k])

    else:
        #Water, Skin, Brain and skull material
        MaterialList['SpecificHeat']=np.zeros_like(MaterialList['SoS'])
        MaterialList['SpecificHeat'][0:3]=[4178,3391,3630]
        if bSegmentedBrain:
             #we add white matter, gray matter and CSF
            MaterialList['SpecificHeat'][3:6]=np.array([3583.0, 3696.0, 4096.0])
            MaterialList['SpecificHeat'][6:]=(1313.0+2274.0)/2
        else:
            MaterialList['SpecificHeat'][3:]=(1313.0+2274.0)/2

        MaterialList['Conductivity']=np.zeros_like(MaterialList['SoS'])
        MaterialList['Conductivity'][0:3]=[0.6,0.37,0.51]
        if bSegmentedBrain:
            MaterialList['Conductivity'][3:6]=np.array([0.55, 0.48, 0.57])
            MaterialList['Conductivity'][6:]=(0.31+0.32)/2
        else:
            MaterialList['Conductivity'][3:]=(0.31+0.32)/2

        MaterialList['Perfusion']=np.zeros_like(MaterialList['SoS'])
        MaterialList['Perfusion'][0:3]=[0,106,559]
        if bSegmentedBrain:
            MaterialList['Perfusion'][3:6]=np.array([764.0, 212.0, 0.0])
            MaterialList['Perfusion'][6:]=(10.0+30.0)/2
        else:
            MaterialList['Perfusion'][3:]=(10.0+30.0)/2

        MaterialList['Absorption']=np.zeros_like(MaterialList['SoS'])
        MaterialList['Absorption'][0:3]=[0,0.85,0.85]
        if bSegmentedBrain:
            MaterialList['Absorption'][3:5]=0.85
            if not bForceNoAbsorptionSkullScalp:
                MaterialList['Absorption'][6:]=(0.16+0.15)/2
        else:
            if not bForceNoAbsorptionSkullScalp:
                MaterialList['Absorption'][3:]=(0.16+0.15)/2

        MaterialList['InitTemperature']=np.ones_like(MaterialList['SoS'])*BaselineTemperature


    SaveDict={}
    SaveDict['MaterialList']=MaterialList
    
    nFactorMonitoring=int(50e-3/dt) # we just track every 50 ms
    if nFactorMonitoring==0:
        nFactorMonitoring=1
    TotalDurationSteps=int((DurationUS+.001)/dt)
    nStepsOn=int(DurationUS/dt) 
    if nFactorMonitoring > TotalDurationSteps:
        nFactorMonitoring=1 #in the weird case TotalDurationSteps is less than 50 ms
    TotalDurationStepsOff=int((DurationOff+.001)/dt)
    TotalDurationBetweenGroups=int(PauseBetweenGroupedSonications/dt)

    xf=Input['x_vec']
    yf=Input['y_vec']
    zf=Input['z_vec']

    LocIJK=Input['TargetLocation'].flatten()
    if bForceHomogenousMedium:
        SelSkull = MaterialMap >0 # we select all material 
        BrainID =[1]
    elif 'MaterialMapCT' in Input:
        if bSegmentedBrain:
            BrainID=[2,3,4,5]
            SelSkull =MaterialMap>=6
            SelSkull =MaterialMap>=3
        else:
            BrainID=[2]
            SelSkull =(MaterialMap>1) & (MaterialMap<4)
    else:
        if bSegmentedBrain:
            BrainID=[4,5,6,7]
        else:
            BrainID=[4]

        SelSkull =(MaterialMap>1) &\
            (MaterialMap<4)

    cx=LocIJK[0]
    cy=LocIJK[1]
    cz=LocIJK[2]

    SelBrain=np.isin(MaterialMap,BrainID)
    SelSkin=MaterialMap==1
 
    if type(InputPData) is str:   
        PressureRatio,RatioLosses=AnalyzeLosses(pAmp,MaterialMap,LocIJK,Input,
                                                MaterialList,pAmpWater,Isppa,
                                                xf,yf,zf,SelBrain,bForceHomogenousMedium,bSegmentedBrain)
    else:
        PressureRatio=np.zeros(len(InputPData),dtype=AllInputs.dtype)
        RatioLosses=np.zeros(len(InputPData),dtype=AllInputs.dtype)
        for n in range(len(InputPData)):
            pAmp=AllInputs[n,:,:,:]
            pAmpWater=AllInputsWater[n,:,:,:]
            print('*'*40)
            print('Calculating losses for spot ',n)
            PressureRatio[n],RatioLosses[n]=AnalyzeLosses(pAmp,MaterialMap,LocIJK,Input,
                                                          MaterialList,pAmpWater,Isppa,
                                                          xf,yf,zf,SelBrain,bForceHomogenousMedium,bSegmentedBrain)
            print('*'*40)
        print('Average (std) of pressure ratio and losses = %f(%f) , %f(%f)' % (np.mean(PressureRatio),np.std(PressureRatio),np.mean(RatioLosses),np.std(RatioLosses)))
            
    
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
                                                        Backend=Backend,
                                                        stableTemp=BaselineTemperature)
    else:
        InputsBHTE=AllInputs.copy()
        for n in range(len(InputPData)):
            InputsBHTE[n,:,:,:]*=PressureRatio[n]
        ResTemp,ResDose,MonitorSlice,Qarr=BHTEMultiplePressureFields(InputsBHTE,
                                                      MaterialMap,
                                                      MaterialList,
                                                      (Input['x_vec'][1]-Input['x_vec'][0]),
                                                      TotalDurationSteps,
                                                      nStepsOnOffList,
                                                      cy,
                                                      nFactorMonitoring=nFactorMonitoring,
                                                      dt=dt,
                                                      Backend=Backend,
                                                      stableTemp=BaselineTemperature)

    ResTempSkin=ResTemp * SelSkin.astype(np.float32)
    ResTempBrain=ResTemp * SelBrain.astype(np.float32)
    ResTempSkull=ResTemp * SelSkull.astype(np.float32)
    mxSkin,mySkin,mzSkin=np.unravel_index(np.argmax(ResTempSkin, axis=None), ResTempSkin.shape)
    mxBrain,myBrain,mzBrain=np.unravel_index(np.argmax(ResTempBrain, axis=None), ResTempBrain.shape)
    mxSkull,mySkull,mzSkull=np.unravel_index(np.argmax(ResTempSkull, axis=None), ResTempSkull.shape)

    MonitoringPointsMap=np.zeros(MaterialMap.shape,np.uint32)
    if bForceHomogenousMedium==1:
        MonitoringPointsMap[mxBrain,myBrain,mzBrain]=1
    else:
        MonitoringPointsMap[mxSkin,mySkin,mzSkin]=1
        MonitoringPointsMap[mxBrain,myBrain,mzBrain]=2
        MonitoringPointsMap[mxSkull,mySkull,mzSkull]=3
        if not(cx==mxBrain and cy==myBrain and cz==mzBrain):
            MonitoringPointsMap[cx,cy,cz]=4
    print('Total # of grouped sonications :',NumberGroupedSonications)
    print('Total # of repetitions in a single group:',Repetitions)
    TOTAL_Iterations = NumberGroupedSonications * Repetitions

    if type(InputPData) is str:
        PMaps=pAmp*PressureRatio
        nStepsOnIn=nStepsOn
    else:
        PMaps=InputsBHTE
        nStepsOnIn=nStepsOnOffList

    FinalTemp=None
    FinalDose=None
    TemperaturePoints=None

    if TOTAL_Iterations <= LimitBHTEIterationsPerProcess:
        nCurrent=0
        ResTemp,ResDose,FinalTemp,FinalDose,TemperaturePoints,nCurrent=RunBHTECycles(nCurrent,
                    Repetitions,
                    TOTAL_Iterations,
                    TotalDurationBetweenGroups,
                    TotalDurationStepsOff,
                    LimitBHTEIterationsPerProcess,
                    InputPData,
                    PMaps,
                    MaterialMap,
                    MaterialList,
                     (Input['x_vec'][1]-Input['x_vec'][0]),
                    TotalDurationSteps,
                    nStepsOn,
                    cy,
                    nFactorMonitoring,
                    dt,
                    DutyCycle,
                    Backend,
                    MonitoringPointsMap,
                    BaselineTemperature,
                    TemperaturePoints,
                    FinalTemp,
                    FinalDose)

    else:
        queueResult=Queue()
        nCurrent=0
        while(nCurrent<TOTAL_Iterations):
            fieldWorkerProcess = Process(target=RunInProcess, 
                                        args=(queueResult,Backend,deviceName,queueMsg,
                                                LimitBHTEIterationsPerProcess,nCurrent,
                                                NumberGroupedSonications,
                                                Repetitions,
                                                InputPData,
                                                PMaps
                                                ,MaterialMap,
                                                MaterialList,
                                                (Input['x_vec'][1]-Input['x_vec'][0]),
                                                TotalDurationSteps,
                                                TotalDurationStepsOff,
                                                nStepsOnIn,cy,nFactorMonitoring,dt,
                                                DutyCycle,MonitoringPointsMap,
                                                TemperaturePoints,BaselineTemperature,
                                                FinalTemp,FinalDose,
                                                TotalDurationBetweenGroups))

            fieldWorkerProcess.start()
            while(True):
                time.sleep(0.1)
                if not queueResult.empty():
                    break
            ProcResults=queueResult.get()
            ResTemp=ProcResults[0]
            ResDose=ProcResults[1]
            FinalTemp=ProcResults[2]
            FinalDose=ProcResults[3]
            TemperaturePoints=ProcResults[4]
            nCurrent=ProcResults[5]
            fieldWorkerProcess.terminate()
            print('process terminated')

    SaveDict['MonitorSlice']=MonitorSlice[:,:,int(nStepsOn/nFactorMonitoring)-1]
    SaveDict['mSkin']=np.array([mxSkin,mySkin,mzSkin]).astype(int)
    SaveDict['mBrain']=np.array([mxBrain,myBrain,mzBrain]).astype(int)
    SaveDict['mSkull']=np.array([mxSkull,mySkull,mzSkull]).astype(int)
    SaveDict['dt']=dt
    if type(InputPData) is str:
        SaveDict['p_map']=pAmp*PressureRatio
        SaveDict['p_map_central']=pAmp[:,cy,:]*PressureRatio
    else:
        SaveDict['p_map']=InputsBHTE.max(axis=0)
        SaveDict['p_map_central']=InputsBHTE.max(axis=0)[:,cy,:]
    SaveDict['MaterialMap_central']=MaterialMap[:,cy,:]
    SaveDict['MaterialMap']=MaterialMap
    SaveDict['PressureRatio']=PressureRatio

    TI=ResTemp[SelBrain].max()
    
    TIS=ResTemp[SelSkin].max()

    TIC=ResTemp[SelSkull].max()
    
    print('Max. Temp. Brain, Max Temp. Skin, Max Temp. Skull',TI,TIS,TIC);

    CEMBrain=FinalDose[SelBrain].max()/60 # in min
    
    CEMSkin=FinalDose[SelSkin].max()/60 # in min

    CEMSkull=FinalDose[SelSkull].max()/60 # in min
    
    print('CEMBrain,CEMSkin,CEMSkull',CEMBrain,CEMSkin,CEMSkull)

    maxPressureLocation=SaveDict['p_map'][SelBrain].argmax()

    MaxBrainPressure = SaveDict['p_map'][SelBrain][maxPressureLocation]
        
    MI=MaxBrainPressure/1e6/np.sqrt(Frequency/1e6)
    MaxIsppa=MaxBrainPressure**2/(2.0*SaveDict['MaterialList']['SoS'][BrainID]*SaveDict['MaterialList']['Density'][BrainID])
    MaxIsppa=MaxIsppa/1e4
    MaxIspta=DutyCycle*MaxIsppa

    Ispta =DutyCycle*Isppa

    SaveDict['MaxBrainPressure']=MaxBrainPressure
    if bForceHomogenousMedium:
        IndTarget=0
    elif (cx==mxBrain and cy==myBrain and cz==mzBrain):
        IndTarget=2
    else:
        IndTarget=3
    SaveDict['TempProfileTarget']=TemperaturePoints[IndTarget,:]
    SaveDict['TimeProfileTarget']=np.arange(SaveDict['TempProfileTarget'].size)*dt
    SaveDict['TemperaturePoints']=TemperaturePoints #these are max points in skin, brain, skull and target
    SaveDict['MI']=MI
    SaveDict['x_vec']=xf*1e3
    SaveDict['y_vec']=yf*1e3
    SaveDict['z_vec']=zf*1e3
    SaveDict['TI']=TI-BaselineTemperature
    SaveDict['TIC']=TIC-BaselineTemperature
    SaveDict['TIS']=TIS-BaselineTemperature
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
    if type(InputPData) is str: 
        for k in ['XSteering','YSteering','ZSteering']:
            if k in Input:
                SaveDict[k]=Input[k]
    else:
        for k in ['XSteering','YSteering','ZSteering']:
            if k in Input:
                steering =np.ones(len(ALL_ACFIELDSKULL))
                for n,entry in enumerate(ALL_ACFIELDSKULL):
                    steering[n]=entry[k]
                SaveDict[k]=steering
    SaveDict['AdjustmentInRAS']=Input['AdjustmentInRAS']
    SaveDict['DistanceFromSkin']=Input['DistanceFromSkin']
    SaveDict['TxMechanicalAdjustmentZ']=Input['TxMechanicalAdjustmentZ']
    SaveDict['TargetLocation']=Input['TargetLocation']
    SaveDict['ZIntoSkinPixels']=Input['ZIntoSkinPixels']
    SaveDict['RatioLosses']=RatioLosses
    SaveDict['DurationUS']=DurationUS
    SaveDict['DurationOff']=DurationOff
    SaveDict['DutyCycle']=DutyCycle
    SaveDict['PRF']=PRF
    SaveDict['BaselineTemperature']=BaselineTemperature
    
    SaveToH5py(SaveDict,outfname+'.h5')
    savemat(outfname+'.mat',SaveDict)
    
    return outfname
        
