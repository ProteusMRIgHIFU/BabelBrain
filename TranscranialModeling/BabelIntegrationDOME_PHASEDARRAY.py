'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - June 28, 2021
     last update   - Nov 28, 2021

'''
from .BabelIntegrationBASE import (RUN_SIM_BASE, 
                            BabelFTD_Simulations_BASE,
                            SimulationConditionsBASE,
                            Material)
import numpy as np
from sys import platform
import os
from stl import mesh
import scipy
from trimesh import creation 
import matplotlib.pyplot as plt
from BabelViscoFDTD.tools.RayleighAndBHTE import ForwardSimple
from .H317 import GenerateH317Tx
import nibabel
from multiprocessing import Process,Queue
from scipy.ndimage import binary_erosion
    
def CreateCircularCoverage(DiameterFocalBeam=1.5e-3,DiameterCoverage=10e-3):
    RadialL=np.arange(DiameterFocalBeam,DiameterCoverage/2,DiameterFocalBeam)
    ListPoints=[[1e-6,0.0]] #center , and we do a trick to be sure all points gets the same treatment below (just make one coordinate different to 0 but very small)
    nEven=0
    for r in RadialL:
        Perimeter=np.pi*r*2
        nSteps=int(Perimeter/DiameterFocalBeam)
        theta=np.arange(nSteps)*np.pi*2/nSteps
        if nEven%2==0:
            theta+=(theta[1]-theta[0])/2
        nEven+=1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        xxyy=np.vstack((x,y)).T
        ListPoints+=xxyy.tolist()
    ListPoints=np.array(ListPoints)
  
    return ListPoints

def CreateSpreadFocus(DiameterFocalBeam=1.5e-3):
    BaseTriangle =  DiameterFocalBeam/2
    HeightTriangle = np.sin(np.pi/3)*DiameterFocalBeam
    ListPoints = [[0,HeightTriangle/2]]
    ListPoints += [[BaseTriangle,-HeightTriangle/2]]
    ListPoints += [[-BaseTriangle,-HeightTriangle/2]]
    ListPoints=np.array(ListPoints)
    return ListPoints


class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self,**kargs):
        return BabelFTD_Simulations(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    DistanceConeToFocus=self._DistanceConeToFocus,
                                     **kargs)
        
    def RunCases(self,
                    XSteering=0.0,
                    YSteering=0.0,
                    ZSteering=0.0,
                    RotationZ=0.0,
                    DistanceConeToFocus=27e-3,
                    MultiPoint=None,
                    **kargs):
        self._RotationZ=RotationZ
        self._DistanceConeToFocus=DistanceConeToFocus
        if MultiPoint is None:
            self._XSteering=XSteering
            self._YSteering=YSteering
            self._ZSteering=ZSteering
            ExtraAdjustX = [XSteering]
            ExtraAdjustY = [YSteering]
            return super().RunCases(ExtraAdjustX=ExtraAdjustX,
                                     ExtraAdjustY=ExtraAdjustY,
                                     **kargs)
        else:
            #we need to expand accordingly to all points
            ExtraAdjustX=[]
            ExtraAdjustY=[]
            for entry in MultiPoint:
                ExtraAdjustX.append(entry['X']+XSteering)
                ExtraAdjustY.append(entry['Y']+YSteering)
            fnames=[]
            for entry in MultiPoint:
                newextrasufffix="_Steer_X_%2.1f_Y_%2.1f_Z_%2.1f_" % (entry['X']*1e3,entry['Y']*1e3,entry['Z']*1e3)
                self._XSteering=entry['X']+XSteering
                self._YSteering=entry['Y']+YSteering
                self._ZSteering=entry['Z']+ZSteering
                fnames+=super().RunCases(extrasuffix=newextrasufffix,
                                         ExtraAdjustX=ExtraAdjustX,
                                         ExtraAdjustY=ExtraAdjustY,
                                         **kargs)     
            
        return fnames

##########################################

class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    #Meta class dealing with the specificis of each test based on the string name
    def __init__(self,
                 XSteering=0.0,
                 YSteering=0.0,
                 ZSteering=0.0,
                 RotationZ=0.0,
                 **kargs):
        
        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._RotationZ=RotationZ
        super().__init__(**kargs)

    def CreateSimConditions(self,**kargs):
        return SimulationConditions(XSteering=self._XSteering,
                                    YSteering=self._YSteering,
                                    ZSteering=self._ZSteering,
                                    RotationZ=self._RotationZ,
                                    Aperture=0.16, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                                    FocalLength=135e-3,
                                    **kargs)

    def AdjustMechanicalSettings(self,SkullMaskDataOrig,voxelS):
        pass

    def GenerateSTLTx(self,prefix):
        #we also export the STL of the Tx for display in Brainsight or 3D slicer
        TxVert=self._SIM_SETTINGS._TxOrig['VertDisplay'].T.copy()
        TxVert/=self._SIM_SETTINGS.SpatialStep
        TxVert=np.vstack([TxVert,np.ones((1,TxVert.shape[1]))])
        affine=self._SkullMask.affine
        
        LocSpot=np.array(np.where(self._SkullMask.get_fdata()==5.0)).flatten()

        TxVert[2,:]=-TxVert[2,:]
        TxVert[0,:]+=LocSpot[0]
        TxVert[1,:]+=LocSpot[1]
        TxVert[2,:]+=LocSpot[2] - self._SIM_SETTINGS._TxMechanicalAdjustmentZ/self._SIM_SETTINGS.SpatialStep

        TxVert=np.dot(affine,TxVert)

        TxStl = mesh.Mesh(np.zeros(self._SIM_SETTINGS._TxOrig['FaceDisplay'].shape[0]*2, dtype=mesh.Mesh.dtype))

        TxVert=TxVert.T[:,:3]
        for i, f in enumerate(self._SIM_SETTINGS._TxOrig['FaceDisplay']):
            TxStl.vectors[i*2][0] = TxVert[f[0],:]
            TxStl.vectors[i*2][1] = TxVert[f[1],:]
            TxStl.vectors[i*2][2] = TxVert[f[3],:]

            TxStl.vectors[i*2+1][0] = TxVert[f[1],:]
            TxStl.vectors[i*2+1][1] = TxVert[f[2],:]
            TxStl.vectors[i*2+1][2] = TxVert[f[3],:]
        
        bdir=os.path.dirname(self._MASKFNAME)
        TxStl.save(bdir+os.sep+prefix+'Tx.stl')
        

    def AddSaveDataSim(self,DataForSim):
        DataForSim['XSteering']=self._XSteering
        DataForSim['YSteering']=self._YSteering
        DataForSim['ZSteering']=self._ZSteering
        DataForSim['RotationZ']=self._RotationZ
        DataForSim['bDoRefocusing']=self._bDoRefocusing
        DataForSim['BasePhasedArrayProgrammingRefocusing']=self._SIM_SETTINGS.BasePhasedArrayProgrammingRefocusing
        DataForSim['BasePhasedArrayProgramming']=self._SIM_SETTINGS.BasePhasedArrayProgramming
    
class SimulationConditions(SimulationConditionsBASE):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,FactorEnlarge = 1.0, #putting a Tx with same F# but just bigger helps to create a more coherent input field for FDTD
                      Aperture=0.30, # m, aperture of the Tx, used tof calculated cross section area entering the domain
                      FocalLength=150e-3,
                      XSteering=0.0, #lateral steering
                      YSteering=0.0,
                      ZSteering=0.0,
                      RotationZ=0.0,#rotation of Tx over Z axis
                      **kargs):
        super().__init__(Aperture=Aperture*FactorEnlarge,FocalLength=FocalLength*FactorEnlarge,**kargs)
        self._FactorEnlarge=FactorEnlarge
        self._OrigAperture=Aperture
        self._OrigFocalLength=FocalLength
        self._Aperture=Aperture*FactorEnlarge
        self._FocalLength=FocalLength*FactorEnlarge
        self._XSteering=XSteering
        self._YSteering=YSteering
        self._ZSteering=ZSteering
        self._RotationZ=RotationZ

    def UpdateConditions(self, SkullMaskNii,AlphaCFL=1.0,bWaterOnly=False,
                         bForceHomogenousMedium=False,
                         BenchmarkTestFile='',
                         DomeType=False):
        super().UpdateConditions(SkullMaskNii,AlphaCFL=AlphaCFL,bWaterOnly=bWaterOnly,
                         bForceHomogenousMedium=bForceHomogenousMedium,
                         BenchmarkTestFile=BenchmarkTestFile,
                         DomeType=True)
        

    def GenTransducerGeom(self):
        raise NotImplementedError("This method should be implemented in the derived class.")
        
    def CalculateRayleighFieldsForward(self,deviceName='6800'):
        print("Precalculating Rayleigh-based field as input for FDTD...")
        #first we generate the high res source of the tx elements
        self.GenTransducerGeom()
        #We replicate as in the GUI as need to account for water pixels there in calculations where to truly put the Tx
        TargetLocation =np.array(np.where(self._SkullMaskDataOrig==5.0)).flatten()
        LineOfSight=self._SkullMaskDataOrig[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()*self._SkullMaskNii.header.get_zooms()[2]/1e3
        print('StartSkin',StartSkin)

        for k in ['center','elemcenter','VertDisplay']:
            self._Tx[k][:,0]+=self._TxMechanicalAdjustmentX
            self._Tx[k][:,1]+=self._TxMechanicalAdjustmentY
            self._Tx[k][:,2]+=self._TxMechanicalAdjustmentZ
            self._TxHighRes[k][:,0]+=self._TxMechanicalAdjustmentX
            self._TxHighRes[k][:,1]+=self._TxMechanicalAdjustmentY
            self._TxHighRes[k][:,2]+=self._TxMechanicalAdjustmentZ

     
        #we apply an homogeneous pressure 
       
        print('min,max Tx Z',self._Tx['center'][:,2].min(),self._Tx['center'][:,2].max())

        cwvnb_extlay=np.array(2*np.pi*self._Frequency/Material['Water'][1]+1j*0).astype(np.complex64)
        
        #we store the phase to reprogram the Tx in water only conditions, required later for real experiments
        self.BasePhasedArrayProgramming=np.zeros(self._Tx['NumberElems'],np.complex64)
        self.BasePhasedArrayProgrammingRefocusing=np.zeros(self._Tx['NumberElems'],np.complex64)

        Amplitude=1.0
        if 'Amplitude1W' in self._Tx:
            Amplitude=self._Tx['Amplitude1W']['Rayleigh']
            print('using 1W Rayleigh per element ampltiude',Amplitude)
        if self._XSteering!=0.0 or self._YSteering!=0.0 or self._ZSteering!=0.0:
            print('Running Steering')
            ds=np.ones((1))*self._SpatialStep**2
        
        
            #we apply an homogeneous pressure 
            u0=np.zeros((1),np.complex64)
            u0[0]=1+0j
            center=np.zeros((1,3),np.float32)
            center[0,0]=self._XDim[self._FocalSpotLocation[0]]+self._TxMechanicalAdjustmentX+self._XSteering
            center[0,1]=self._YDim[self._FocalSpotLocation[1]]+self._TxMechanicalAdjustmentY+self._YSteering
            center[0,2]=self._ZDim[self._FocalSpotLocation[2]]+self._TxMechanicalAdjustmentZ+self._ZSteering

            print('center',center)
            
            u2back=ForwardSimple(cwvnb_extlay,center,ds.astype(np.float32),u0,self._Tx['elemcenter'].astype(np.float32),deviceMetal=deviceName)
            u0=np.zeros((self._Tx['center'].shape[0],1),np.complex64)
            nBase=0
            for n in range(self._Tx['NumberElems']):
                phi=np.angle(np.conjugate(u2back[n]))
                self.BasePhasedArrayProgramming[n]=np.conjugate(u2back[n])
                u0[nBase:nBase+self._Tx['elemdims']]=np.exp(1j*phi).astype(np.complex64)
                nBase+=self._Tx['elemdims']
        else:
             u0=(np.ones((self._Tx['center'].shape[0],1),np.float32)+ 1j*np.zeros((self._Tx['center'].shape[0],1),np.float32))
        nxf=len(self._XDim)
        nyf=len(self._YDim)
        nzf=len(self._ZDim)
        yp,xp,zp=np.meshgrid(self._YDim,self._XDim,self._ZDim)
        
        rf=np.hstack((np.reshape(xp,(nxf*nyf*nzf,1)),np.reshape(yp,(nxf*nyf*nzf,1)), np.reshape(zp,(nxf*nyf*nzf,1)))).astype(np.float32)
        
        u0*= self.AdjustWeightAmplitudes()*Amplitude

        u2=ForwardSimple(cwvnb_extlay,self._Tx['center'].astype(np.float32),self._Tx['ds'].astype(np.float32),u0,rf,deviceMetal=deviceName)
        u2=np.reshape(u2,xp.shape)*1.5e6 # in Pa
        
        self._u2RayleighField=u2

        
    def CreateSources(self,ramp_length=8):
        #we create the list of functions sources taken from the Rayliegh incident field
        LengthSource=np.floor(self._TimeSimulation/(1.0/self._Frequency))*1/self._Frequency
        TimeVectorSource=np.arange(0,LengthSource+self._TemporalStep,self._TemporalStep)
        #we do as in k-wave to create a ramped signal
        
        ramp_length_points = int(np.round(ramp_length/self._Frequency/self._TemporalStep))
        ramp_axis =np.arange(0,np.pi,np.pi/ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points=len(ramp)
        
        self._SourceMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)

        nBase=0
        nBaseVert=0
        Orig=[self._XDim[0],self._YDim[0],self._ZDim[0]]

        PulseSource = np.zeros((self._TxHighRes['NumberElems'],TimeVectorSource.shape[0]))

        AmplitudeCal=1.0
        if 'Amplitude1W' in self._Tx:
            AmplitudeCal=self._Tx['Amplitude1W']['Visco'][self._Frequency][self._basePPW]
            print('Using amplitude for 1W',AmplitudeCal)

        for n in range(self._TxHighRes['NumberElems']):
            SelCenters=self._TxHighRes['center'][nBase:nBase+self._TxHighRes['elemdims'],:]
            SelCenters=np.vstack((self._TxHighRes['center'][nBase:nBase+self._TxHighRes['elemdims'],:],
                                self._TxHighRes['VertDisplay'][nBaseVert:nBase+self._TxHighRes['elemdims']*4,:]))
            
            IndX=np.round((SelCenters[:,0]-Orig[0])/self._SpatialStep).astype(int)
            IndY=np.round((SelCenters[:,1]-Orig[1])/self._SpatialStep).astype(int)
            IndZ=np.round((SelCenters[:,2]-Orig[2])/self._SpatialStep).astype(int)
            assert(np.all(IndX>=self._PMLThickness))
            assert(np.all(IndX<(self._N1-self._PMLThickness)))
            assert(np.all(IndY>=self._PMLThickness))
            assert(np.all(IndY<(self._N2-self._PMLThickness)))
            assert(np.all(IndZ>=self._PMLThickness))
            assert(np.all(IndZ<(self._N3-self._PMLThickness)))
            assert(np.all(self._SourceMap[IndX,IndY,IndZ]==0))
            self._SourceMap[IndX,IndY,IndZ]=n+1

            nBase+=self._TxHighRes['elemdims']
            nBaseVert+=self._TxHighRes['elemdims']*4

            PulseSource[n,:] = np.sin(2*np.pi*self._Frequency*TimeVectorSource+np.angle(self.BasePhasedArrayProgramming[n]))*AmplitudeCal
            PulseSource[n,:int(ramp_length_points)]*=ramp
            PulseSource[n,-int(ramp_length_points):]*=np.flip(ramp)

            
        self._PulseSource=PulseSource
        self._PulseAmplitude=AmplitudeCal
        
        ## Now we create the sources for back propagation
        
        self._PunctualSource=np.sin(2*np.pi*self._Frequency*TimeVectorSource).reshape(1,len(TimeVectorSource))
        self._PunctualSource[0,:int(ramp_length_points)]*=ramp
        self._PunctualSource[0,-int(ramp_length_points):]*=np.flip(ramp)
        
        self._SourceMapPunctual=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        LocForRefocusing=self._FocalSpotLocation.copy()
        LocForRefocusing[0]+=int(np.round(self._XSteering/self._SpatialStep))
        LocForRefocusing[1]+=int(np.round(self._YSteering/self._SpatialStep))
        LocForRefocusing[2]+=int(np.round(self._ZSteering/self._SpatialStep))
        self._SourceMapPunctual[LocForRefocusing[0],LocForRefocusing[1],LocForRefocusing[2]]=1

    def CreateSensorMap(self):
        '''
        Create the sensor map and back-propagation sensor map for the simulation.
        '''
        self._SensorMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        # for the back propagation, we only use the entering face
        self._SensorMapBackPropagation=np.zeros((self._N1,self._N2,self._N3),np.uint32)    
    
        self._SensorMap[self._PMLThickness:-self._PMLThickness,
                        self._PMLThickness:-self._PMLThickness,
                        self._ZSourceLocation+1:-self._PMLThickness]=1
        
        self._SensorMapBackPropagation=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        Orig=[self._XDim[0],self._YDim[0],self._ZDim[0]]
        self._IndexSensorsBack=[]
        for n in range(self._Tx['NumberElems']):
            center=self._Tx['elemcenter'][n,:]
            IndX=np.round((center[0]-Orig[0])/self._SpatialStep).astype(int)
            IndY=np.round((center[1]-Orig[1])/self._SpatialStep).astype(int)
            IndZ=np.round((center[2]-Orig[2])/self._SpatialStep).astype(int)
            self._SensorMapBackPropagation[IndX,IndY,IndZ]=1
            self._IndexSensorsBack.append((IndX,IndY,IndZ))


        
    def CalculatePhaseData(self,bRefocused=False,bDoRefocusing=True,bDoRefocusingVolume=False):
        #we overwrite to use a volume
        super().CalculatePhaseData(bRefocused=bRefocused,bDoRefocusing=bDoRefocusing,bDoRefocusingVolume=True)
        
    def BackPropagationRayleigh(self,deviceName='6800'):
        for n in range(self._Tx['NumberElems']):
            IndX,IndY,IndZ=self._IndexSensorsBack[n]
            u2back=self._PressMapFourierBack[IndX,IndY,IndZ]
            self.BasePhasedArrayProgrammingRefocusing[n]=np.conjugate(u2back)
            
        
    def CreateSourcesRefocus(self,ramp_length=8):
        #we create the list of functions sources taken from the Rayliegh incident field
        LengthSource=np.floor(self._TimeSimulation/(1.0/self._Frequency))*1/self._Frequency
        TimeVectorSource=np.arange(0,LengthSource+self._TemporalStep,self._TemporalStep)
        #we do as in k-wave to create a ramped signal
        
        ramp_length_points = int(np.round(ramp_length/self._Frequency/self._TemporalStep))
        ramp_axis =np.arange(0,np.pi,np.pi/ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points=len(ramp)

        PulseSource = np.zeros((self._TxHighRes['NumberElems'],TimeVectorSource.shape[0]))

        for n in range(self._TxHighRes['NumberElems']):

            PulseSource[n,:] = np.sin(2*np.pi*self._Frequency*TimeVectorSource+np.angle(self.BasePhasedArrayProgrammingRefocusing[n]))*self._PulseAmplitude
            PulseSource[n,:int(ramp_length_points)]*=ramp
            PulseSource[n,-int(ramp_length_points):]*=np.flip(ramp)
        
        self._PulseSourceRefocus=PulseSource
         
    def RUN_SIMULATION(self,bDoStressSource=False,SelRMSorPeak=1,bApplyCorrectionForDispersion=True,**kargs):
        super().RUN_SIMULATION(bDoStressSource=True,bApplyCorrectionForDispersion=False,SelRMSorPeak=2,**kargs)
